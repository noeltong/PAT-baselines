from run_lib import train, eval
import os
from absl import app, flags
from ml_collections.config_flags import config_flags
import warnings

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory to store files.")
flags.DEFINE_enum("mode", None, ['train', 'eval', 'tune'], "Running mode: train, eval or finetune.")
flags.mark_flags_as_required(['mode', 'config'])


def main(argv):
    warnings.filterwarnings('ignore')

    config = FLAGS.config

    if FLAGS.workdir is not None:
        work_dir = os.path.join('workspace', FLAGS.workdir)
    else:
        name_idx = [config.model.arch, config.data.mask, 'batch' + str(config.training.batch_size), 'known' + str(config.data.num_known)]
        work_dir = os.path.join('workspace', f"run_{'_'.join(name_idx)}")

    if FLAGS.mode == 'train':
        train(args=config, work_dir=work_dir)
    elif FLAGS.mode == 'eval':
        eval(args=config, work_dir=work_dir)


if __name__ == '__main__':
    app.run(main)
