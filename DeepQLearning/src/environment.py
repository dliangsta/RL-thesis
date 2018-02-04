import gym
import random
import logging
import numpy as np

try:
    import scipy.misc
    imresize = scipy.misc.imresize
    imwrite = scipy.misc.imsave
except:
    import cv2
    imresize = cv2.resize
    imwrite = cv2.imwrite
    

class AtariEnvironment(object):
    """
    The atari environment for a reinforcement learning agent.
    """
    
    def __init__(self, env_name, n_action_repeat, max_random_start,  observation_dims, display, use_cumulated_reward):
        # Make the gym environment.
        self.env = gym.make(env_name)
        # How many times to repeat an action.
        self.n_action_repeat = n_action_repeat
        # Maximum number of no-ops when starting new game.
        self.max_random_start = max_random_start
        # Size of action space.
        self.action_size = self.env.action_space.n
        # If display should be rendered.
        self.display = display
        # Dimensions of observation space.
        self.observation_dims = observation_dims
        # If cumulative reward should be used, or simply last reward during action repeat.
        self.use_cumulated_reward = use_cumulated_reward
        # Count number of games.
        self.count = 0

    def new_game(self, from_random_game=False):
        """
        Start a new game.
        """
        screen = self.env.reset()
        screen, reward, terminal, _ = self.env.step(0)
        self.count = 0

        if self.display:
            self.env.render()

        if from_random_game:
            return screen, 0, False
        else:
            self.lives = self.env.unwrapped.ale.lives()
            terminal = False
            return self.preprocess(screen), 0, terminal

    def new_random_game(self):
        """
        Start a new game, some random number of no-op steps into the game.
        """
        screen, reward, terminal = self.new_game(True)

        for idx in range(random.randrange(self.max_random_start)):
            screen, reward, terminal, _ = self.env.step(0)

        if self.display:
            self.env.render()

        self.lives = self.env.unwrapped.ale.lives()

        terminal = False

        return self.preprocess(screen), 0, terminal

    def step(self, action, is_training):
        """
        Execute the specified action in the environment.
        """
        if action == -1:
            # Step with random action.
            action = self.env.action_space.sample()

        cumulated_reward = 0

        for _ in range(self.n_action_repeat):
            # Take step.
            screen, reward, terminal, _ = self.env.step(action)
            # Accumulate reward.
            cumulated_reward += reward
            # Update lives.
            current_lives = self.env.unwrapped.ale.lives()

            if reward:
                self.count += 1

            if is_training and self.lives > current_lives:
                terminal = True

            if terminal:
                break

        if self.display:
            self.env.render()

        if not terminal:
            self.lives = current_lives

        if terminal:
            terminal = self.count
            self.count = 0

        if self.use_cumulated_reward:
            return self.preprocess(screen), cumulated_reward, terminal, {}
        else:
            return self.preprocess(screen), reward, terminal, {}

    def preprocess(self, raw_screen):
        """
        Preprocess, fixing weird screen coloring and resizing.
        """
        y = 0.2126 * raw_screen[:, :, 0] + 0.7152 * raw_screen[:, :, 1] + 0.0722 * raw_screen[:, :, 2]
        y = y.astype(np.uint8)
        y_screen = imresize(y, self.observation_dims)
        return y_screen
