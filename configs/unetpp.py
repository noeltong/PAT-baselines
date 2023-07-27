from configs.default_config import get_config as get_default_config


def get_config():

    cfg = get_default_config()

    # ----------------
    # Train
    # ----------------

    training = cfg.training
    training.num_epochs = 100
    training.batch_size = 64

    # ----------------
    # Model
    # ----------------

    model = cfg.model
    model.clip_grad_norm = 1.
    model.arch = 'UNet_PP'

    # ----------------
    # Optimization
    # ----------------

    optim = cfg.optim
    optim.initial_lr = 5e-4
    optim.warmup_epochs = None

    # ----------------
    # Data
    # ----------------

    data = cfg.data
    data.data_dir = '/root/data/mice/npy/train'
    data.num_known = 32
    data.mask = 'uniform_mask'

    cfg.seed = 42
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg