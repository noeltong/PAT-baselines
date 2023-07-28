import logging
import os
import torch
from torch import nn
from torch.nn import functional as F
from utils.data import load_data, get_mask_fn
from utils.time import time_calculator
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from utils.optim import get_optim
from utils.utils import seed_everything
from torch.utils.tensorboard import SummaryWriter
from utils.utils import AverageMeter
from models.model_utils import get_arch
from models.loss import get_loss_fn
from utils.PAT import DASSparseOperator
from utils.utils import min_max_scaler
from utils.eval import get_metric_fn


def train(args, work_dir):
    """Runs the training pipeline.

    Args:
    config: ml_collections.ConfigDict(), config of the project
    workdir: directory to store files.
    """

    torch.backends.cudnn.benchmark = True
    workdir = os.path.join(work_dir, 'train', args.data.mask, str(args.data.num_known))

    # -------------------
    # Initialize DDP
    # -------------------

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # -------------------
    # seeds
    # -------------------

    seed_everything(args.seed + rank)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # -----------------------------
    # Create directories for data
    # -----------------------------

    log_dir = os.path.join(workdir, 'tensorboard')
    ckpt_dir = os.path.join(workdir, 'ckpt')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------
    # Loggers
    # -------------------

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s')

    fh = logging.FileHandler(os.path.join(
        workdir, 'train_log.log'), encoding='utf-8')
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)

    if rank == 0:
        logger.info(f'Work files stored in {workdir}.')

    # -------------------
    # Load data
    # -------------------

    if rank == 0:
        logger.info('Loading transforms...')

    DAS = DASSparseOperator(args)

    if rank == 0:
        logger.info('Transform function loaded.')

    if rank == 0:
        logger.info('Loading data...')

    train_loader, train_sampler, test_loader, test_sampler = load_data(args)

    if rank == 0:
        logger.info(f'Data loaded.')

    dist.barrier()

    # -------------------
    # Initialize model
    # -------------------

    if rank == 0:
        logger.info('Begin model initialization...')

    model = get_arch(args.model.arch)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        logger.info("Models initialized.")

    dist.barrier()

    # -------------------
    # define optimization
    # -------------------

    if rank == 0:
        logger.info('Handling optimizations...')

    optimizer, scheduler = get_optim(model, args)
    criterion = get_loss_fn(args)

    if rank == 0:
        logger.info('Completed.')

    # -------------------
    # training loop
    # -------------------

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    time_logger = time_calculator()
    iters_per_epoch = len(train_loader)
    mask_fn = get_mask_fn(args)
    get_metric = get_metric_fn(args)

    dist.barrier()
    torch.cuda.empty_cache()

    for epoch in range(args.training.num_epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        model.train()
        train_loss_epoch = AverageMeter()

        if rank == 0:
            logger.info(f'Start training epoch {epoch + 1}.')

        # ----------------------------
        # run the training process
        # ----------------------------

        for i, (x, y) in enumerate(train_loader):
            x, y = x.float().cuda(non_blocking=True), y.float().cuda(non_blocking=True)
            x = DAS.signal_to_image(mask_fn(x))

            # rescale data to [-1, 1] for faster convergence
            if args.training.rescale == True:
                x = min_max_scaler(x) / 0.5 - 1
                y = min_max_scaler(y) / 0.5 - 1

            with torch.cuda.amp.autocast(enabled=True):
                out = model(x)
                loss = criterion(out, y)

            train_loss_epoch.update(loss.item(), x.shape[0])
            if rank == 0:
                writer.add_scalar("Train/Loss", train_loss_epoch.val,
                                  epoch * iters_per_epoch + i)

            logger.info(
                f'Epoch: {epoch + 1}/{args.training.num_epochs}, Iter: {i + 1}/{iters_per_epoch}, Loss: {train_loss_epoch.val:.6f}, Device: {rank}')

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if args.model.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.model.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        dist.barrier()
        if rank == 0:
            logger.info(
                f'Epoch: {epoch + 1}/{args.training.num_epochs}, Avg loss: {train_loss_epoch.avg:.4f}, Time: {time_logger.time_length()}')

        # save snapshot periodically
        if (epoch + 1) % args.training.save_ckpt_freq == 0:
            states = model.state_dict()
            if rank == 0:
                logger.info(f'Saving snapshot at epoch {epoch + 1}')
                torch.save(states, os.path.join(
                    ckpt_dir, f'{epoch+1}_loss_{train_loss_epoch.avg:.2f}.pth'))

        # Report loss on eval dataset periodically
        if (epoch + 1) % args.training.eval_freq == 0:
            if rank == 0:
                logger.info(f'Start evaluate at epoch {epoch + 1}.')

            with torch.no_grad():
                model.eval()
                iters_per_eval = len(test_loader)
                eval_loss_epoch = AverageMeter()
                eval_psnr_epoch = AverageMeter()
                eval_ssim_epoch = AverageMeter()

                for i, (x, y) in enumerate(test_loader):
                    x, y = x.float().cuda(non_blocking=True), y.float().cuda(non_blocking=True)
                    x = DAS.signal_to_image(mask_fn(x))

                    if args.training.rescale == True:
                        x = min_max_scaler(x) / 0.5 - 1
                        y = min_max_scaler(y) / 0.5 - 1
                        
                    with torch.cuda.amp.autocast(enabled=True):
                        out = model(x)
                        loss = criterion(out, y)

                    for n in range(x.shape[0]):
                        p, s = get_metric(min_max_scaler(out[n, ...].float().squeeze()), min_max_scaler(y[n, ...].squeeze()))
                        eval_psnr_epoch.update(p)
                        eval_ssim_epoch.update(s)
                    

                    eval_loss_epoch.update(loss.item(), x.shape[0])
                    logger.info(
                        f'Epoch: {epoch + 1}/{args.training.num_epochs}, Iter: {i + 1}/{iters_per_eval}, Loss: {eval_loss_epoch.val:.6f}, Time: {time_logger.time_length()}, PSNR: {eval_psnr_epoch.val}, SSIM: {eval_ssim_epoch.val}, Device: {rank}')

                if rank == 0:
                    writer.add_scalar('Eval/Eval loss', eval_loss_epoch.avg, epoch)
                    writer.add_scalar('Eval/PSNR', eval_psnr_epoch.avg, epoch)
                    writer.add_scalar('Eval/SSIM', eval_ssim_epoch.avg, epoch)

                if rank == 0:
                    logger.info(
                        f'Epoch: {epoch + 1}/{args.training.num_epochs}, Avg eval loss: {eval_loss_epoch.avg:.4f}, PSNR: {eval_psnr_epoch.avg}, SSIM: {eval_ssim_epoch.avg}.')

        dist.barrier()

    states = model.state_dict()
    if rank == 0:
        logger.info(
            f'Training complete. Total time: {time_logger.time_length()}.')
        torch.save(states, os.path.join(ckpt_dir, 'final.pth'))


def eval(args, work_dir):
    """Runs the evaluation pipeline.

    Args:
    config: ml_collections.ConfigDict(), config of the project
    workdir: directory to store files.
    """

    torch.backends.cudnn.benchmark = True

    # -----------------------------
    # Create directories for data
    # -----------------------------

    ckpt_dir = os.path.join(workdir, 'train', args.data.mask, str(args.data.num_known), 'ckpt')
    workdir = os.path.join(work_dir, 'eval', args.data.mask, str(args.data.num_known))

    # -------------------
    # Initialize DDP
    # -------------------

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # -------------------
    # seeds
    # -------------------

    seed_everything(args.seed + rank)

    if args.use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    # -------------------
    # Loggers
    # -------------------

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s')

    fh = logging.FileHandler(os.path.join(
        workdir, 'train_log.log'), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # -------------------
    # Load data
    # -------------------

    if rank == 0:
        logger.info('Loading transforms...')

    DAS = DASSparseOperator(args)

    if rank == 0:
        logger.info('Transform function loaded.')
    
    # -------------------
    # Initialize model
    # -------------------

    if rank == 0:
        logger.info('Begin model initialization...')

    model = get_arch(args.model.arch)
    model.load_state_dict(
        torch.load(args.ckpt, map_location='cpu')
    )
    model = model.cuda()

    if rank == 0:
        logger.info("Models initialized.")

    dist.barrier()

    def get_data(args, path):
        data = np.load(path)
        sinogram = data['sinogram']
        gt = data['gt']

        mask = get_mask_fn(args)
        masked = mask * sinogram
        masked = torch.from_numpy(masked)
        noisy = from_space(masked)

        return noisy


    paths = glob(os.path.join(args.data.data_dir, "*.npz"))
    for path in paths:
        
        with torch.inference_mode():
            out = model