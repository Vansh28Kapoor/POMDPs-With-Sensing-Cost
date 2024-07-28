import numpy as np
#import pulp
import math
import argparse
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
from planner import valueEvaluation, Q_pi, brute_force_search
from Heuristic import Solve, V_blind

if __name__ == "__main__":
    actions = ["R", "B"]
    sensingActions = [action + "S" for action in actions]
    numHeadStates = 2
    windowLength = 4
    alpha = 0.5
    sensingcost = 0.25
    states = generate_states(numHeadStates, actions, windowLength)
    # states contains all paths in the tree 
    
    # T = {
    #     "R": np.array([[0.28,0.72],[0.934,0.066]]),
    #     "B": np.array([[0.31,0.69],[0.481,0.519]])
    # }

    # C = {
    #     "R": np.array([0.066,0.502]),
    #     "B": np.array([0.29,0.41])
    # }

    # T = {
    #     "R": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.7, 0.1]]),
    #     "B": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.7, 0.1]]),
    #     "G": np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.7, 0.1]])
    # }

    # C = {
    #     "R": np.array([0, 1, 0.5]),
    #     "B": np.array([1, 0, 0.5]),
    #     "G": np.array([0, 0, 0.5])
    # }

    # T = {
    #     "R": np.array([[0.28,0.72],[0.934,0.066]]),
    #     "B": np.array([[0.31,0.69],[0.481,0.519]])
    # }

    # C = {
    #     "R": np.array([0.066,0.502]),
    #     "B": np.array([0.29,0.41])
    # }

    T = {
        "R": np.array([[0.7, 0.3], [0.2, 0.8]]),
        "B": np.array([[0.3, 0.7], [0.9, 0.1]])
    }

    C = {
        "R": np.array([0.25, 0.75]),
        "B": np.array([1, 0.5])
    }


    # for action in actions:
    #     print(T[action], C[action])
    mdp = generate_pomdp(windowLength, T, C, alpha, actions, numHeadStates, sensingcost)

    # policy, value_function  = brute_force_search(states, actions+sensingActions, mdp, 0.9, windowLength)
    with open('mdp.txt', 'w') as f:
        f.write(str(mdp))

    opt_policy, opt_val = brute_force_search(states, actions+sensingActions, mdp, alpha, windowLength)
    policy = {i: tuple([opt_policy[tuple([i])]]) for i in range(numHeadStates)}
    with open('opt_policy.txt', 'w') as f:
        f.write(str(opt_policy))
    for i in range(numHeadStates):
        while(policy[i][-1][-1] != 'S'):
            policy[i] = policy[i] + tuple([opt_policy[tuple([i])+policy[i]]])

    for i in range(numHeadStates):
        print(f"state {i}, policy: {policy[i]}, value: {opt_val[tuple([i])]}")
        