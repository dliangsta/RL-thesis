import random
import numpy as np


class Experience(object):
    """
    An experience that an agent will have. Contains information like actions, observations, and rewards.
    """

    def __init__(self, batch_size, history_length, memory_size, observation_dims):
        self.batch_size = batch_size
        self.history_length = history_length
        self.memory_size = memory_size

        # Elements of an experience: actions, rewards ,observations, and terminals.
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.observations = np.empty([self.memory_size] + observation_dims, dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        # Pre-allocate prestates and poststates for minibatch.
        self.prestates = np.empty([self.batch_size, self.history_length] + observation_dims, dtype=np.float16)
        self.poststates = np.empty([self.batch_size, self.history_length] + observation_dims, dtype=np.float16)

        self.count = 0
        self.current = 0

    def add(self, observation, reward, action, terminal):
        """
        Add an observation, reward, and action to the experiences.
        """
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.observations[self.current, ...] = observation
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        """
        Take a random sample of the experiences.
        """
        indices = []
        while len(indices) < self.batch_size:
            # Find a random index.
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if index >= self.current and index - self.history_length < self.current:
                    continue
                if self.terminals[(index - self.history_length):index].any():
                    continue
                break
            
            # Retrieve prestate and poststates.
            self.prestates[len(indices), ...] = self.retreive(index - 1)
            self.poststates[len(indices), ...] = self.retreive(index)

            # Record index as used.
            indices.append(index)

        # Gather actions, rewards, and terminals for indices.
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
            rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals

    def retreive(self, index):
        """
        Retrieve a specified observation.
        """
        index = index % self.count
        if index >= self.history_length - 1:
            return self.observations[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            indices = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.observations[indices, ...]
