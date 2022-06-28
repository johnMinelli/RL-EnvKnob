from argparse import ArgumentParser, ArgumentTypeError, ArgumentDefaultsHelpFormatter
from warnings import warn
from utils.utils import create_path
from path import Path
from os import path



def positive_int(value: any) -> int:
    """
    Checks if a value is a positive integer.

    :param value: the value to be checked.
    :return: the value if valid integer, otherwise raises an ArgumentTypeError.
    """
    int_value = int(value)

    if int_value <= 0:
        raise ArgumentTypeError("%s should be a positive integer value." % value)

    return int_value


def positive_float(value: any) -> float:
    """
    Checks if a value is a positive float.

    :param value: the value to be checked.
    :return: the value if valid float, otherwise raises an ArgumentTypeError.
    """
    float_value = float(value)

    if float_value <= 0:
        raise ArgumentTypeError("%s should be a positive float value." % value)

    return float_value

class Options():
    def __init__(self):
        self.parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # basic info
        self.parser.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment. It decides where to store results and models')
        self.parser.add_argument('--seed', type=int, default=0, help='Seed for random functions, and network initialization')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # model
        self.parser.add_argument('-s', '--save_path', type=str, default='./save', help='Checkpoints are saved here')
        self.parser.add_argument('-as', '--agent_save_interval', type=positive_int, default=100, required=False, help='The save interval for the trained agent (default %(default)s), in episodes.')
        self.parser.add_argument('--load_path', type=str, required=False, default=None, help='Path where to search the trained agents to be loaded (default `save_path`/models.')
        self.parser.add_argument('--load_solver_agent', type=int, required=False, default=None, help='Label of a trained solver agent to be loaded from `load_path` (default %(default)s). -1 to load the last saved model in the folder.')
        self.parser.add_argument('--load_generator_agent', type=int, required=False, default=None, help='Label of a trained generator agent to be loaded from `load_path` (default %(default)s). -1 to load the last saved model in the folder.')

        # output
        self.parser.add_argument('-op', '--out_path', type=str, default='./out', help='Results are saved here')
        self.parser.add_argument('-pf', '--print_freq', type=int, default=50, help='Frequency of showing training results on console')
        self.parser.add_argument('-t', '--tensorboard', action='store_true', help='Log stats on tensorboard local dashboard')
        self.parser.add_argument('-w', '--wandb', action='store_true', help='Log stats on wandb dashboard')
        self.parser.add_argument('--sweep_id', type=str, help='Sweep id for wandb hyperparameters search')
        self.parser.add_argument('-r', '--render_mode', default='human', required=False, type=str.lower, choices=['human', 'rgb_array', None], help='Modality of rendering of the environment.')
        self.parser.add_argument('-rec', '--record', required=False, action='store_true', help='Whether the game should be recorded. Please note that you need to have ffmpeg in your path!')


        self.parser.add_argument('--cnn', type=positive_int, default=0, required=False, help='Use CNN (MobileNetV2) instead of FC.')
        self.parser.add_argument('--frame_skipping', type=positive_int, default=8, required=False, help='The frames to skip per action (default %(default)s).')
        self.parser.add_argument('--alternate_training_interval', type=positive_int, default=10, required=False, help='The episodes to run before alternate training between solver and generator (default %(default)s).')
        self.parser.add_argument('-e', '--episodes', type=positive_int, default=1000, required=False, help='The episodes to run the training procedure (default %(default)s).')
        self.parser.add_argument('-b', '--batch_size', type=positive_int, default=128, required=False, help='The batch size to be sampled from the memory for the training (default %(default)s).')
        self.parser.add_argument('-k', '--K_epochs', type=positive_int, default=10, required=False, help='The number of epochs to run on the single batch (default %(default)s).')
        self.parser.add_argument('--step_reward', type=float, default=0, required=False, help='The (negative) reward to assign for each step (default %(default)s).')
        self.parser.add_argument('--adv_gamma', type=positive_float, default=0.99, required=False, help='The discount factor of PPO advantage (default %(default)s).')
        self.parser.add_argument('--adv_lambda', type=positive_float, default=0.95, required=False, help='The discount factor of PPO advantage (default %(default)s).')
        self.parser.add_argument('-opt', '--optimizer', type=str.lower, default='sgd', required=False, choices=['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'], help='The optimizer to be used. (default %(default)s).')
        self.parser.add_argument('--lr_a', type=positive_float, default=float(0.00001), required=False, help='The learning rate for the actor optimizer (default %(default)s).')
        self.parser.add_argument('--lr_c', type=positive_float, default=float(0.00001), required=False, help='The learning rate for the critic optimizer (default %(default)s).')
        self.parser.add_argument('--lr_decay', action='store_true', required=False, help='Enable learning rate decay for the optimizer.')
        self.parser.add_argument('--lr_beta1', type=positive_float, default=0.9, required=False, help='The beta 1 for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_beta2', type=positive_float, default=0.999, required=False, help='The beta 2 for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_rho', type=positive_float, default=0.95, required=False, help='The rho for the optimizer (default %(default)s).')
        self.parser.add_argument('--lr_fuzz', type=positive_float, default=0.01, required=False, help='The fuzz factor for the "rmsprop" optimizer (default %(default)s).')
        self.parser.add_argument('--lr_momentum', type=positive_float, default=0.1, required=False, help='The momentum for the "sgd" optimizer (default %(default)s).')

        # log
        self.parser.add_argument('-li', '--log_interval', type=positive_int, default=20, required=False,
                            help='The current scoring information interval (default %(default)s), in episodes.')
        self.parser.add_argument('-iim', '--info_interval_mean', type=positive_int, default=100, required=False,
                            help='The mean scoring information interval (default %(default)s), in episodes.')
        self.parser.add_argument('-tci', '--target_change_interval', type=positive_int, default=int(1E4), required=False,
                            help='The target model change interval (default %(default)s), in steps.')
        self.parser.add_argument('-ah', '--agent_history', type=positive_int, required=False, default=4,
                            help='The agent\'s frame history (default %(default)s).')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        self._check_args_consistency()

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt

    def _check_args_consistency(self):
        """ Checks the input arguments. """
        # Set default variables.
        poor_observe = bad_target_model_change = 500
        frame_history_ceiling = 10

        # Create the path to the files, if necessary.
        self.opt.models_path = Path(self.opt.save_path)/"models/"
        self.opt.plots_path = Path(self.opt.out_path)/self.opt.name/"plots/"
        self.opt.results_path = Path(self.opt.out_path)/self.opt.name/"results/"

        create_path(self.opt.models_path)
        create_path(self.opt.plots_path)
        create_path(self.opt.results_path)

        if self.opt.load_path is None:
            self.opt.load_path = self.opt.models_path

        if self.opt.info_interval_mean == 1:
            warn('Info interval mean has no point to be 1. '
                 'The program will continue, but the means will be ignored.'.format(self.opt.info_interval_mean))

        if self.opt.target_change_interval < bad_target_model_change:
            warn('Target model change is extremely small ({}). This will possibly make the agent unstable.'
                 'Consider a value greater than {}'.format(self.opt.target_change_interval, bad_target_model_change))

        if self.opt.agent_history > frame_history_ceiling:
            warn(
                'The agent\'s frame history is too big ({}). This will possibly make the agent unstable and slower.'
                'Consider a value smaller than {}'.format(self.opt.agent_history, frame_history_ceiling))

        # # Downsampling should result with at least 32 pixels on each dimension,
        # # because the first convolutional layer has a filter 8x8 with stride 4x4.
        # if not frame_can_pass_the_net(game.observation_space_shape[1], game.observation_space_shape[2]):
        #     raise ValueError('Downsample is too big. It can be set from 1 to {}'.format(
        #         min(int(game.pixel_rows / MIN_FRAME_DIM_THAT_PASSES_NET),
        #             int(game.pixel_columns / MIN_FRAME_DIM_THAT_PASSES_NET))))

        # final_memory_size = agent.memory.end + self.opt.observe
        # if final_memory_size < self.opt.batch_size:
        #     raise ValueError('The total number of observing steps ({}) '
        #                      'cannot be smaller than the agent\'s memory size ( current = {}, final = {} )'
        #                      ' after the observing steps ({}).'.format(self.opt.observe, agent.memory.end,
        #                                                                final_memory_size, self.opt.observe))
