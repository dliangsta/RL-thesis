import gym
import numpy as np
import pickle
from tqdm import trange

class Agent():

    def __init__(self, run_dir, save_dir):
        self.run_dir = run_dir
        self.save_dir = save_dir

        self.env = gym.make('FrozenLake8x8-v0')
        self.Q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.learning_rate = .8
        self.discount_rate = .99
        self.t = 0
        self.t_max = 1000000
        self.test_freq = 1000
        self.test_max = 1000
        self.test_len = 30
        self.mean_rewards = []

        self.performance_log_filename = save_dir + 'data/performance_log.csv'
        self.pkl_filename = save_dir + 'data/agent.pkl'

    def train(self):
        t = trange(self.t, self.t_max, desc='rl-thesis', ncols=200)
        for self.t in t:
            # Train agent.
            self.play_episode(True)

            if self.t % self.test_freq == 0:
                # Run for ___ episodes.
                rewards = []
                for game in range(self.test_max):
                    reward = self.play_episode(False)
                    rewards.append(reward)

                # Statistics.
                self.mean_rewards.append(np.mean(rewards))
                if len(self.mean_rewards) >= self.test_len:
                    recent_avg = np.mean(self.mean_rewards[-self.test_len:])
                    recent_var = np.var(self.mean_rewards[-self.test_len:])
                else:
                    recent_avg = 0.
                    recent_var = 0.
                    
                status_string = 'iteration: %9d, avg: %.3f, recent_avg: %.9f, recent_var: %.9f, learning_rate: %.9f' % (self.t, np.mean(rewards), recent_avg, recent_var, self.learning_rate)
                open(self.performance_log_filename,'a').write(status_string.replace(':',',') + '\n')

                # Temporary stopping criterion.
                if np.mean(rewards) >= .99:
                    print('Done!')

                # Set description in TQDM bar.
                t.set_description(status_string)

                # Save.
                self.save()
                print()

    def play_episode(self, train):
        # Reset environment and get first new observation
        state = self.env.reset()
        reward = 0
        
        # The Q_table-Table learning algorithm
        for step in range(1000):

            if train:
                # Choose an action by greedily (with noise) picking from Q table
                action = np.argmax(self.Q_table[state, :] + np.random.randn(1, self.env.action_space.n) * 1000. / (self.t + 1))
            else:
                # Choose best action from Q table
                action = np.argmax(self.Q_table[state, :])

            # Get new state and reward from environment
            state_next, r, done, _ = self.env.step(action)

            if train:
                # Update Q_table-Table with new knowledge
                self.Q_table[state, action] = self.Q_table[state, action] + self.learning_rate * (r + self.discount_rate * np.max(self.Q_table[state_next, :]) - self.Q_table[state, action])
        
            reward += r
            state = state_next
        
            if done == True:
                break

        self.learning_rate = max(.08, .99999 * self.learning_rate)
        return reward

    def save(self):
        pickle.dump(self, open(self.pkl_filename,'wb'))
    