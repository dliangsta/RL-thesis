import gym
import os
import time
import random
import numpy as np
import tensorflow as tf
import subprocess
from statsmodels.stats.proportion import proportion_confint

from time import time, ctime
from tqdm import trange

from .history import History
from .experience import Experience


class DeepQAgent(object):
    """
    Agent equipped with a DQN.
    """
    def __init__(self, sess, pred_network, env, stat, config, target_network=None):
        # TensorFlow session.
        self.sess = sess
        # Statistics module.
        self.stat = stat
        # Iteration count.
        self.t = 0

        # Config flags.
        self.is_train = config.is_train
        self.ep_start = config.ep_start
        self.ep_end = config.ep_end
        self.history_length = config.history_length
        self.t_ep_end = config.t_ep_end
        self.t_learn_start = config.t_learn_start
        self.t_train_freq = config.t_train_freq
        self.t_target_q_update_freq = config.t_target_q_update_freq
        self.t_test = config.t_test
        self.env_name = config.env_name
        self.discount_r = config.discount_r
        self.min_r = config.min_r
        self.max_r = config.max_r
        self.observation_dims = config.observation_dims
        self.learning_rate = config.learning_rate
        self.learning_rate_minimum = config.learning_rate_minimum
        self.learning_rate_decay = config.learning_rate_decay
        self.learning_rate_decay_step = config.learning_rate_decay_step
        self.load = config.load
        self.chtc = config.chtc
        self.window_length = config.window_length
        self.termination_p_hat = config.termination_p_hat

        # Prediction network.
        self.pred_network = pred_network
        # Target network.
        self.target_network = target_network
        # Operation for copying target network to prediction network.
        self.target_network.create_copy_op(self.pred_network)
        # Atari environment.
        self.env = env
        # History of most recently seen frames, used for determining next action.
        self.history = History(config.batch_size, config.history_length, config.observation_dims)
        # Experience replay of recent frames, rewards, etc.
        self.experience = Experience(config.batch_size, config.history_length, config.memory_size, config.observation_dims)
        # Used for progress bar.
        self.t_range = None

        if config.random_start:
            self.new_game = self.env.new_random_game
        else:
            self.new_game = self.env.new_game

        # Set up optimizer.
        with tf.variable_scope('optimizer'):
            # Target q-values.
            self.targets = tf.placeholder('float32', [None], name='target_q_t')
            # Actions.
            self.actions = tf.placeholder('int64', [None], name='action')
            # One-hot representation of actions.
            actions_one_hot = tf.one_hot(self.actions, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            # Q-values of actions.
            pred_q = tf.reduce_sum(self.pred_network.outputs * actions_one_hot, reduction_indices=1, name='q_acted')
            # Difference between target q values and predicted q-values.
            self.delta = self.targets - pred_q
            # Clipped error.
            self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                          0.5 * tf.square(self.delta),
                                          tf.abs(self.delta) - 0.5, name='clipped_error')
            # Loss.
            self.loss = tf.reduce_mean(self.clipped_error, name='loss')
            # Learning rate.
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.stat.t_op,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))
            # Optimizer.
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    def train(self, t_max):
        """
        Train the network for t_max iterations.
        """
        # Initialize session, model, variables.
        tf.global_variables_initializer().run()
        self.stat.load_model()
        self.target_network.run_copy()
        self.stat.t_start = self.stat.get_t()

        # Burn in.
        self.burn_in()

        print(" [*] Training.")

        # Progress display mechanism.
        if self.chtc:
            self.t_range = range(self.stat.t_start, t_max)
        else:
            self.t_range = trange(self.stat.t_start, t_max)

        # Start a new game.
        observation, reward, terminal = self.new_game()

        # Initialize history.
        for _ in range(self.history_length):
            self.history.add(observation)

        try:
            for self.t in self.t_range:
                # Linearly decaying exploration factor.
                epsilon = self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.t_ep_end - max(0., self.t - self.t_learn_start)) / self.t_ep_end)

                # 1. Predict.
                action = self.predict(self.history.get(), epsilon)
                # 2. Act.
                observation, reward, terminal, _ = self.env.step(action, is_training=True)
                # 3. Observe.
                self.observe(observation, reward, action, terminal)
                # 4. Update.
                _, _, is_update = self.update()
                # 5. Test. 
                terminal = self.test() or terminal

                # Notify the statistic module of the new iteration number, in case it needs to save the model.
                self.stat.on_step(self.t, is_update)
                    
                # If the game has terminated, reset.
                if terminal:
                    observation, reward, terminal = self.new_game()
                    for _ in range(self.history_length):
                        self.history.add(observation)

        except KeyboardInterrupt:
            print("\n [!] Keyboard interrupt registered. Exiting!")
            # The model is typically saved every t_test iterations, but if the training needs to be paused, we can save immediately before quitting.
            self.stat.save_model(self.t, self.stat.latest_saver)

        except Exception as e:
            print(" [!] Unhandled exception encountered:", e, "\nExiting!")
            self.stat.save_model(self.t, self.stat.latest_saver)


        self.stat.zip_data(False)

    def burn_in(self):
        """
        Burn in the experience.
        """
        print(" [*] Burning in.")
        # Progress display mechanism.
        if self.chtc:
            self.t_range = range(0, self.t_learn_start)
        else:
            self.t_range = trange(0, self.t_learn_start)

        # Start a new game.
        observation, reward, terminal = self.new_game()

        # Initialize history.
        for _ in range(self.history_length):
            self.history.add(observation)

        for self.t in self.t_range:
            # Linearly decaying exploration factor.
            epsilon = self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.t_ep_end - max(0., self.t - self.t_learn_start)) / self.t_ep_end)

            # 1. Predict.
            action = self.predict(self.history.get(), epsilon)
            # 2. Act.
            observation, reward, terminal, _ = self.env.step(action, is_training=True)
            # 3. Observe.
            self.observe(observation, reward, action, terminal)

            # If the game has terminated, reset.
            if terminal:
                observation, reward, terminal = self.new_game()
                for _ in range(self.history_length):
                    self.history.add(observation)

    def play(self, test_ep=0., n_step=10000, n_episode=1000):
        """
        Allow the agent to interact in the environment without updating the network.
        """

        # If not training, clear log of old data, initialize TF variables, and load model.
        if self.stat and not self.is_train:
            tf.initialize_all_variables().run()
            self.stat.load_model()

        self.target_network.run_copy()

        rewards = []
        game_lengths = []

        # Play at least n_episode episodes.
        while np.sum(game_lengths) < n_episode:

            # Start a new game.
            observation, reward, terminal = self.new_game()
            current_reward = 0

            # Add initial frames to history.
            for _ in range(self.history_length):
                self.history.add(observation)

            # Play game until 'terminal.'
            for t in range(n_step):
                # 1. Predict.
                action = self.predict(self.history.get(), test_ep)
                # 2. Act.
                observation, reward, terminal, _ = self.env.step(action, is_training=False)
                # 3. Observe.
                self.history.add(observation)
                
                current_reward += reward

                if terminal:
                    break

                # Set tqdm range description.
                if self.t_range and not self.chtc:
                    self.t_range.set_description('PLAY: %d/%d' % (np.sum(game_lengths), n_episode))

            # After game, add game length and rewards.
            rewards.append(float(current_reward))
            game_lengths.append(terminal)

        if self.t_range and not self.chtc:
            self.t_range.set_description()
        
        self.compute_statistics(rewards, game_lengths)


    def observe(self, observation, reward, action, terminal):
        """
        Store a recent observation of screen captures, reward, and action.
        """
        reward = max(self.min_r, min(self.max_r, reward))
        # Add observed data to history and experience.
        self.history.add(observation)
        self.experience.add(observation, reward, action, terminal)

    def update(self):
        """
        Update network if needed.
        """
        result = [], 0, False

        if self.t % self.t_train_freq == 0:
            result = self.q_learning_minibatch()

        if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
            # Copy 
            self.update_target_q_network()

        return result

    def test(self):
        """
        Tests agent.
        """
        if self.t % self.t_test == 0:
            # Every 10 test periods, play for 100 episodes instead of 10 to get a better idea of ability of the agent.
            if self.t % (self.t_test * 10) == 0 and not self.chtc:
                self.play(test_ep=0., n_step=10000, n_episode=100)
            else:
                self.play(test_ep=0., n_step=10000, n_episode=1)
            return True

        return False

    def q_learning_minibatch(self):
        """
        Train over a minibatch.
        """
        if self.experience.count <= self.history_length:
            # Cannot train because not enough experiences.
            return [], 0, False
        else:
            # Sample experience.
            s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

        terminal = np.array(terminal) + 0.

        # Predicted and target q-values for time t+1.
        max_q_t_plus_1 = self.target_network.eval_max_outputs(s_t_plus_1)
        target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward

        # Run training ops.
        _, q_t, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
            self.targets: target_q_t,
            self.actions: action,
            self.pred_network.inputs: s_t,
        })

        return q_t, loss, True

    def compute_statistics(self, rewards, game_lengths):
        """
        Compute statistics and print and log them.
        """
        rewards_mean = np.mean(rewards)
        n = np.sum(game_lengths)
        rewards_sum = np.sum(rewards)

        # Extended rewards: an array of 1's for wins and 0's for losses. Used to compute more statistics.
        extended_rewards = np.append([0. for i in range(int((n - rewards_sum) / 2)) ], [1. for i in range(int((n + rewards_sum) / 2))])
        p_hat = np.mean(extended_rewards)
        s_hat = np.var(extended_rewards)

        # Confidence interval on proportion of won games.
        confint = proportion_confint(np.sum(extended_rewards), len(extended_rewards))

        # Print and log status.
        final_status_string = "t: %9d, rewards: %.3f, p_hat: %.9f, s_hat: %.9f, interval: [%.9f, %.9f], # games: %4d, time: %s, datetime: %s" \
                % (self.t, rewards_mean, p_hat, s_hat, confint[0], confint[1], n, str(time()), ctime())
        print(final_status_string)
        self.stat.write_log(self.stat.log_filename, final_status_string.replace(': ',', ').replace('[','').replace(']','') + "\n")

        if not self.chtc:
            # Upload log to Google Drive for remote viewing.    
            self.stat.upload_log(self.stat.log_filename)
        
        # Zip data.
        self.stat.zip_data(True)

        if self.chtc:
            self.stat.evaluate_termination_criteria()
        
    def predict(self, s_t, epsilon):
        """
        Perform epsilon-greedy action.
        """
        if random.random() < epsilon:
            # Random action.
            action = random.randrange(self.env.action_size)
        else:
            # Best action.
            action = self.pred_network.eval_actions([s_t])[0]

        return action

    def update_target_q_network(self):
        """
        Copy target network to prediction network, done every so often for prediction stability.
        """
        assert self.target_network != None
        self.target_network.run_copy()