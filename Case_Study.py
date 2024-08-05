import numpy as np
# import pulp
import math
import argparse
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
from planner import valueEvaluation, Q_pi, brute_force_search
from Heuristic import Solve, V_blind

if __name__ == "__main__":
    actions = ["0", "1", "2", "3"]
    sensingActions = [action + "S" for action in actions]
    numHeadStates = 4
    windowLength = 0
    states = generate_states(numHeadStates, actions, windowLength)
    print(f'Window_len = {windowLength}')
    T = {
        "0": np.array([[1., 0., 0., 0.],
                       [1., 0., 0., 0.],
                       [0.5, 0.5, 0., 0.],
                       [0., 0.5, 0.5, 0.]]),
        "1": np.array([[1., 0., 0., 0.],
                       [0.5, 0.5, 0., 0.],
                       [0., 0.5, 0.5, 0.],
                       [0., 0., 0.5, 0.5]]),
        "2": np.array([[0.5, 0.5, 0., 0.],
                       [0., 0.5, 0.5, 0.],
                       [0., 0., 0.5, 0.5],
                       [0., 0., 0., 1.]]),
        "3": np.array([[0., 0.5, 0.5, 0.],
                       [0., 0., 0.5, 0.5],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.]])
    }

    C = {
        "0": np.array([0,   -2.,   -2.75, -2.25]),
        "1": np.array([-1,   -1.75, -1.25, -0.75]),
        "2": np.array([-0.75, -0.25,  0.25,  0.75]),
        "3": np.array([0.75,  1.25,  1.75,  2.25])
    }

    for action in actions:
        print(T[action], C[action])
    mdp = generate_pomdp(windowLength, T, C, 0.8, actions,
                         numHeadStates, 0.25)  # Change Sensing Cost

    # policy, value_function  = brute_force_search(states, actions+sensingActions, mdp, 0.9, windowLength)
    # print(mdp)

    opt_policy, opt_val = brute_force_search(
        states, actions+sensingActions, mdp, 0.8, windowLength)
    policy = {i: tuple([opt_policy[tuple([i])]]) for i in range(numHeadStates)}
    # print(opt_policy)
    for i in range(numHeadStates):
        while (policy[i][-1][-1] != 'S'):
            policy[i] = policy[i] + tuple([opt_policy[tuple([i])+policy[i]]])

    for i in range(numHeadStates):
        print(f"state {i}, policy: {policy[i]}, value: {opt_val[tuple([i])]}")
    print(opt_val.values())
