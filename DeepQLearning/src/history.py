import numpy as np


class History:
    """
    Stores history of screen captures.
    """
    
    def __init__(self, batch_size, history_length, screen_dims):
        self.history = np.zeros([history_length] + screen_dims, dtype=np.float32)

    def add(self, screen):
        """
        Add a screen state to the history.
        """
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        """
        Reset the history.
        """
        self.history *= 0

    def get(self):
        """
        Get the history.
        """
        return np.transpose(self.history, (1, 2, 0))
