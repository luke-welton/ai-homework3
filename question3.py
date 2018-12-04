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

env = gym.make("MountainCar-v0")

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


funcs = locals()

for cool_alpha in [True, False]:
    for cool_gamma in [True, False]:
        for cool_epsilon in [True, False]:
            cool_info = "(Cool $\\alpha$: {}, Cool $\gamma$: {}, Cool $\epsilon$: {})".format(
                "Y" if cool_alpha else "N", "Y" if cool_gamma else "N", "Y" if cool_epsilon else "N"
            )

            means = dict()
            variances = dict()

            for v in [means, variances]:
                for alg in ALGS:
                    v[alg] = []

            print(cool_info)
            for r in range(20):
                print("\nRun {}".format(r))
                plot_means = dict()
                plot_vars = dict()

                for alg in ALGS:
                    episode_counts = []

                    initialize_values()
                    current_vals = {
                        "alpha": ALPHA,
                        "gamma": GAMMA,
                        "epsilon": EPSILON
                    }

                    for i in range(1000):
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

                            if cool_alpha:
                                current_vals["alpha"] *= (1 - COOLING / 50)
                            if cool_gamma:
                                current_vals["gamma"] *= (1 - COOLING / 200)
                            if cool_epsilon:
                                current_vals["epsilon"] *= (1 - COOLING)

                        # print("Run {}: Finished on Episode {}".format(i, t))

                        episode_counts.append(t)

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

                    plot_means[alg] = current_means
                    plot_vars[alg] = current_variances

                    mean = current_means[len(current_means) - 1]
                    variance = current_variances[len(current_variances) - 1]

                    means[alg].append(mean)
                    variances[alg].append(variance)

                    print("Mean for {}: {}".format(alg, mean))
                    print("Variance for {}: {}".format(alg, variance))

                env.close()

                if r == 0:
                    for vals in [plot_means, plot_vars]:
                        fig, ax = plt.subplots()

                        title = ""
                        if vals == plot_means:
                            title = "Mean over Iterations"
                        else:
                            title = "Variance over Iterations"

                        title += "\n" + cool_info

                        plt.title(title)
                        plt.xticks(np.arange(0, len(vals["Q"]), 100))

                        for key in vals:
                            plt.plot(np.arange(len(vals[key])), vals[key], label=key)

                        plt.legend(loc="best")
                        plt.show()

            for v in [means, variances]:
                fig, ax = plt.subplots()

                title = ""
                if v == means:
                    title = "Mean over Runs"
                else:
                    title = "Variance over Runs"

                title += "\n" + cool_info

                plt.title(title)
                plt.xticks(np.arange(0, 20, 4))

                for key in v:
                    plt.plot(np.arange(20), v[key], label=key)

                plt.legend(loc="best")
                plt.show()
