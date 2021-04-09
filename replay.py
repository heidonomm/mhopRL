from collections import deque, namedtuple
import numpy as np
import random

Experience = namedtuple("Experience", ('state', 'action',
                                       'reward', 'next_state', 'done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append(
            (state, action, reward, next_state, done))

    def sample(self, batch_size, rho=0):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
        # return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def can_provide_sample(self, batch_size):
        return len(self.buffer) >= batch_size
