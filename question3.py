import gym
import sys
import numpy as np
from random import random

try:
    ALPHA = float(sys.argv[1])
    GAMMA = float(sys.argv[2])
    EPSILON = float(sys.argv[3])
except IndexError:
    print("Not enough arguments passed. Shutting down.")
    exit(0)

POSITIONS = np.arange(-12, 7) / 10
VELOCITIES = np.arange(-7, 8) / 100
ACTIONS = np.arange(3)

env = gym.make("MountainCar-v0").env

q_values = dict()
for pair in np.array(np.meshgrid(POSITIONS, VELOCITIES)).T.reshape(-1, 2):
    q_values[tuple(pair)] = dict()


def q(current_state, next_state, reward):
    next_action = max(q_values[next_state].keys(), key=(lambda key: q_values[next_state][key]))

    old_value = q_values[current_state][action]
    new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * q_values[next_state][next_action])
    q_values[current_state][action] = new_value


def sarsa(current_state, next_state, reward):
    if random() < EPSILON:
        next_action = env.action_space.sample()
    else:
        next_action = max(q_values[next_state].keys(), key=(lambda key: q_values[next_state][key]))

    old_value = q_values[current_state][action]
    new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * q_values[next_state][next_action])
    q_values[current_state][action] = new_value


def expected_sarsa(current_state, next_state, reward):
    expectation = EPSILON * env.action_space.sample() \
        + (1 - EPSILON) * max(q_values[next_state].keys(), key=(lambda key: q_values[next_state][key]))

    old_value = q_values[current_state][action]
    new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * expectation)
    q_values[current_state][action] = new_value


def initialize_values():
    for entry in q_values:
        q_values[entry].clear()
        for _action in ACTIONS:
            q_values[entry][_action] = 0


for alg in [q, sarsa, expected_sarsa]:
    initialize_values()

    for i in range(20):
        state = env.reset()
        t = 0
        done = False
        while not done:
            t += 1
            env.render()

            state = tuple(state)
            d = round(state[0], 1)
            v = round(state[1], 2)
            state = (d, v)

            if random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = max(q_values[state].keys(), key=(lambda key: q_values[state][key]))

            next_state, reward, done, info = env.step(action)

            next_state_t = tuple(next_state)
            next_d = round(next_state_t[0], 1)
            next_v = round(next_state_t[1], 2)
            next_state_t = (next_d, next_v)

            alg(state, next_state_t, reward)

            state = next_state
        print("Run {}: Finished on Episode {}".format(i, t))
    print("\n")

env.close()
