import gym
import random
import tensorflow as tf
import os

from pprint import PrettyPrinter

# Define flags for constants.
flags = tf.app.flags

# Running/loading related flags.
flags.DEFINE_string('run_dir', './', 'Directory that DQN is run in.')
flags.DEFINE_string('save_dir', './', 'Directory to store model in.')
flags.DEFINE_boolean('load', True, 'Whether to load model or not.')
flags.DEFINE_boolean('chtc', False, 'Whether running on CHTC or not.')
flags.DEFINE_integer('max_to_keep', 0, 'Max number of checkpoints to keep (0 for unlimited).')

# Environment related flags.
flags.DEFINE_string('env_name', 'Pong-v0','The name of gym environment to use.')
flags.DEFINE_integer('n_action_repeat', 1, 'The number of actions to repeat.')
flags.DEFINE_integer('max_random_start', 30,'The maximum number of NOOP actions at the beginning of an episode.')
flags.DEFINE_integer('history_length', 4,'The length of history of observation to use as an input to DQN.')
flags.DEFINE_integer('max_r', +1, 'The maximum value of clipped reward.')
flags.DEFINE_integer('min_r', -1, 'The minimum value of clipped reward.')
flags.DEFINE_string('observation_dims','[80, 80]', 'The dimension of gym observation.')
flags.DEFINE_boolean('random_start', 'True','Whether to start with random state.')
flags.DEFINE_boolean('use_cumulated_reward', False,'Whether to use cumulated reward or not.')


# Flags that will be scaled.
flags.DEFINE_integer('scale', 1000, 'The scale for big numbers.')
flags.DEFINE_integer('memory_size', 1000, 'The size of experience memory (*= scale).')
flags.DEFINE_integer('t_target_q_update_freq', 10, 'The frequency of target network to be updated (*= scale).')
flags.DEFINE_integer('t_ep_end', 1000, 'The time when epsilon reach ep_end (*= scale).')
flags.DEFINE_integer('t_train_max', 250000,'The maximum number of t while training (*= scale).')
flags.DEFINE_integer('learning_rate_decay_step', 100, 'The learning rate of training (*= scale).')
flags.DEFINE_integer('t_learn_start', 100, 'The time when to begin training (*= scale).')
flags.DEFINE_integer('t_test', 100, 'Number of steps until test.')
flags.DEFINE_integer('t_save', 1000, 'Number of steps until save.')
flags.DEFINE_integer('n_step', 10, 'Number of steps per episode while playing.')
flags.DEFINE_integer('n_episode', 1, 'Number of episodes while running.')

# Training flags.
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing.')
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy.')
flags.DEFINE_float('ep_end', 0.1, 'The value of epsilon at the end in e-greedy.')
flags.DEFINE_float('discount_r', 0.99, 'The discount factor for reward.')
flags.DEFINE_integer('t_train_freq', 4, 'Frequency of train operation.')
flags.DEFINE_integer('batch_size', 32, 'The size of batch for minibatch training.')
flags.DEFINE_float('learning_rate', 0.00025, 'The learning rate of training.')
flags.DEFINE_float('learning_rate_minimum', 0.0002, 'The decay of learning rate of training.')
flags.DEFINE_float('learning_rate_decay', 0.96, 'The decay of learning rate of training.')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer.')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return.')
flags.DEFINE_float('beta', 0.01, 'Beta of RMSProp optimizer.')

# Statistics flags.
flags.DEFINE_integer('window_length', 100, 'Size of window over which the score average used for termination criteria is computed.')
flags.DEFINE_float('termination_p_hat', .78, 'Value of terminating value for average of sliding window of p_hat.')

# Miscellaneous flags.
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not.')
flags.DEFINE_string('log_level', 'INFO','Log level from [DEBUG, INFO, WARNING, ERROR, CRITICAL].')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed.')

config = flags.FLAGS

from src.agent import DeepQAgent

# Set random seed.
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

# Set TF logging level.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main(_):
    config.observation_dims = eval(config.observation_dims)

    # Scale some of the flags.
    for flag in ['memory_size', 't_target_q_update_freq', 't_ep_end', 't_train_max', 'learning_rate_decay_step', 't_learn_start', 't_test', 't_save', 'n_step', 'n_episode']:
        setattr(config, flag, getattr(config, flag) * config.scale)

    # Determine some more flags, clean up flags.
    config.chtc = os.path.abspath(config.run_dir) != os.path.abspath(config.save_dir)
    if config.chtc:
        config.t_test /= 10
    config.max_to_keep = 0 if not config.chtc else 2
    config.run_dir = config.run_dir.replace('//','/')
    config.save_dir = config.save_dir.replace('//','/')

    # Print config.
    PrettyPrinter().pprint({key: config.__dict__['__wrapped'][key].value for key in config.__dict__['__wrapped'].__dir__()})        

if __name__ == '__main__':
    tf.app.run()
