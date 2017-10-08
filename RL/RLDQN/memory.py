from collections import deque
import numpy as np


# TODO:implement prioritized Replay Memory

class RandomMemory:
    # Random Replay Memory
    def __init__(self, max_size=256):
        self.memory = deque(maxlen=max_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size=64):
        if len(self.memory) < batch_size:
            raise Exception("Memory's length is smaller than batch size")
        else:
            ids = np.random.choice(len(self.memory), batch_size)
            return [self.memory[id] for id in ids]
