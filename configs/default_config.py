from ml_collections.config_dict import ConfigDict


def get_config():

    cfg = ConfigDict()

    # ----------------
    # Train
    # ----------------

    training = cfg.training = ConfigDict()
    training.num_epochs = 200
    training.batch_size = 32
    training.save_ckpt_freq = 50
    training.eval_freq = 10
    training.rescale = True

    # ----------------
    # Model
    # ----------------

    model = cfg.model = ConfigDict()
    model.clip_grad_norm = 1.
    model.arch = 'UNet_3P'

    # ----------------
    # Optimization
    # ----------------

    cfg.optim = optim = ConfigDict()
    optim.optimizer = 'RAdam'
    optim.schedule = 'CosineAnnealingLR'
    optim.loss = 'MSELoss'
    optim.initial_lr = 0.0005
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = None

    # ----------------
    # Data
    # ----------------

    cfg.data = data = ConfigDict()
    data.num_workers = 2
    data.prefetch_factor = 1
    data.data_dir = '/root/data/mice/npy/train'
    data.num_known = 32
    data.mask = 'random_mask'
    data.resolution = 128
    data.len_sig = 1000
    data.num_sig = 128

    cfg.seed = 42
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg