import numpy as np
import random


class Agent:
    def __init__(self, actions, q_value_model, eps=.1):
        self.actions = actions  # available action
        self.q_value = q_value_model
        self.eps = eps

    def move(self, state):
        action_values = self.q_value.predict([state])
        if random.random() > self.eps:
            return np.argmax(action_values)
        else:
            return np.random.choice(self.actions)

    def train(self, states, targets):
        return self.q_value.train(states, targets)

    def predict(self, states):
        return self.q_value.predict(states)
