import gym
import copy
import Qvalue, agent, memory
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def run(env,
        batch_size, agent, memory, discount, steps=200):
    state = env.reset()
    done = False
    for i in range(steps):
        if done:
            break
        action = agent.move(state)
        next_state, reward, done, _ = env.step(action)
        memory.add((state, action, next_state, reward, done))

        if len(memory.memory) > batch_size:
            print("length:", len(memory.memory))
            state_m, action_m, next_state_m, reward_m, done_m = zip(*memory.sample(batch_size))
            state_m = np.array(state_m)
            action_m = np.array(action_m)
            next_state_m = np.array(next_state_m)
            reward_m = np.array(reward_m)
            done_m = np.array(done_m)

            q_m = agent.predict(next_state_m)

            actual_target_m = reward_m + (1. - done_m) * discount * np.amax(q_m, axis=1)

            targets = agent.predict(state_m)

            # assign the actual reward to the taken action
            for i, action in enumerate(action_m):
                targets[i, action] = actual_target_m[i]
                loss = agent.train(states=state_m, targets=targets)
            state = copy.copy(next_state)


env = gym.make("MountainCar-v0")

n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]
print("n_actions:", n_actions, "state_dim", state_dim)
batch_size = 64
qvalue_model = Qvalue.Qvalue(state_dim=state_dim, n_actions=n_actions, batch_size=64, h1_n=512, h2_n=256)
agent = agent.Agent(actions=n_actions, q_value_model=qvalue_model, eps=.1)
memory = memory.RandomMemory(max_size=1024)

discount = .95
for episode_i in range(1000):
    print("episode_i:", episode_i)
    run(env,
        batch_size, agent, memory, discount, steps=200)
