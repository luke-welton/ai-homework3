import gym
import sys
import numpy as np
from random import random
import matplotlib.pyplot as plt

try:
    ALPHA = float(sys.argv[1])
    GAMMA = float(sys.argv[2])
    EPSILON = float(sys.argv[3])
    COOLING = float(sys.argv[4])
except IndexError:
    print("Not enough arguments passed. Shutting down.")
    exit(0)

ALGS = ["Q", "SARSA", "EXPECTED_SARSA"]

POSITIONS = np.arange(-12, 6) / 10
VELOCITIES = np.arange(-7, 8) / 100
ACTIONS = np.arange(3)

env = gym.make("MountainCar-v0").env

q_values = dict()
for pair in np.array(np.meshgrid(POSITIONS, VELOCITIES)).T.reshape(-1, 2):
    q_values[tuple(pair)] = dict()


def q(current_state, next_state, reward, vals):
    next_action = max(q_values[next_state].keys(), key=(lambda key: q_values[next_state][key]))

    old_value = q_values[current_state][action]
    new_value = (1 - vals["alpha"]) * old_value + \
                vals["alpha"] * (reward + vals["gamma"] * q_values[next_state][next_action])
    q_values[current_state][action] = new_value


def sarsa(current_state, next_state, reward, vals):
    if random() < vals["epsilon"]:
        next_action = env.action_space.sample()
    else:
        next_action = max(q_values[next_state].keys(), key=(lambda key: q_values[next_state][key]))

    old_value = q_values[current_state][action]
    new_value = (1 - vals["alpha"]) * old_value + \
                vals["alpha"] * (reward + vals["gamma"] * q_values[next_state][next_action])
    q_values[current_state][action] = new_value


def expected_sarsa(current_state, next_state, reward, vals):
    expectation = vals["epsilon"] * env.action_space.sample() + \
                  (1 - vals["epsilon"]) * max(q_values[next_state].keys(), key=(lambda key: q_values[next_state][key]))
    next_action = round(expectation)

    old_value = q_values[current_state][action]
    new_value = (1 - vals["alpha"]) * old_value + \
                vals["alpha"] * (reward + vals["gamma"] * q_values[next_state][next_action])
    q_values[current_state][action] = new_value


def initialize_values():
    for entry in q_values:
        q_values[entry].clear()
        for _action in ACTIONS:
            q_values[entry][_action] = 0


means = dict()
variances = dict()
funcs = locals()

for alg in ALGS:
    episode_counts = []

    initialize_values()
    current_vals = {
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon": EPSILON
    }

    for i in range(20):
        # current_vals["gamma"] = GAMMA

        state = env.reset()
        t = 0

        done = False
        while not done:
            t += 1

            state = tuple(state)
            d = round(state[0], 1)
            v = round(state[1], 2)
            state = (d, v)

            if random() < current_vals["epsilon"]:
                action = env.action_space.sample()
            else:
                action = max(q_values[state].keys(), key=(lambda key: q_values[state][key]))

            next_state, reward, done, info = env.step(action)

            next_state_t = tuple(next_state)
            next_d = round(next_state_t[0], 1)
            next_v = round(next_state_t[1], 2)
            next_state_t = (next_d, next_v)

            funcs[alg.lower()](state, next_state_t, reward, current_vals)

            state = next_state

            for key in current_vals:
                current_vals[key] *= (1 - COOLING)

            # current_vals["alpha"] *= (1 - COOLING / 100)
            # current_vals["gamma"] *= (1 - COOLING / 100)
            # current_vals["epsilon"] *= (1 - COOLING / 100)

        print("Run {}: Finished on Episode {}".format(i, t))

        episode_counts.append(t)
    print()

    current_means = []
    current_variances = []
    total_sum = 0
    for i, count in enumerate(episode_counts):
        total_sum += count
        current_means.append(total_sum / (i + 1))

        variance = 0
        for j in range(i + 1):
            variance += (episode_counts[j] - current_means[i]) ** 2
        variance /= (i + 1)
        current_variances.append(variance)

    means[alg] = current_means
    variances[alg] = current_variances

    mean = current_means[len(current_means) - 1]
    variance = current_variances[len(current_variances) - 1]

    print("Mean for {}: {}".format(alg, mean))
    print("Variance for {}: {}".format(alg, variance))
    print("\n")

env.close()

for vals in [means, variances]:
    fig, ax = plt.subplots()

    if vals == means:
        plt.title("Mean over Iterations")
    else:
        plt.title("Variance over Iterations")

    plt.xticks(np.arange(0, 21, 4))

    for key in vals:
        print(key)
        plt.plot(np.arange(20), vals[key], label=key)

    plt.legend(loc="upper right")
    plt.show()
