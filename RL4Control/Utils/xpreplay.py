from collections import namedtuple
from collections import deque
import numpy as np
from numpy.random import default_rng


class xpreplay():
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)
        self.curr_sequence = namedtuple('Sequence', ['state', 'action', 'reward', 'n_state'])
        self.num_inbuffer = min(self.batch_size, len(self.history))
        self.rng = default_rng(123)
        

    def store_sequence(self, state, action, reward, n_state):
        sequ = self.curr_sequence(state, action, reward, n_state)
        self.history.append(sequ)

    def sample_batch(self):
        #indexes = np.random.randint(0, self.batch_size + 1, size = (self.batch_size))
        indexes = self.rng.choice(self.buffer_size, size=self.batch_size, replace=False)
        #print(indexes)
        batch = list([self.history[i] for i in indexes])
        states = [h.state for h in batch]
        actions = np.array([h.action for h in batch])
        rewards = np.array([h.reward for h in batch])
        n_states = [h.n_state for h in batch]

        return states, actions, rewards, n_states


    