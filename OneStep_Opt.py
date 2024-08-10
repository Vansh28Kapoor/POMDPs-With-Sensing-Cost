import numpy as np
# import pulp
import math
import argparse
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
from planner import valueEvaluation, Q_pi, brute_force_search
from Heuristic import Solve, V_blind
from copy import deepcopy


def Z(window_length, T, C, numHeadStates, gamma):  # Returns Z(i) & Beliefs for lowermost states
    states = generate_states(numHeadStates, actions, windowLength+1)

    Belief = {tuple([i]): np.array([1 if i == j else 0 for j in range(
        numHeadStates)]) for i in range(numHeadStates)}
    Z = {tuple([i]): 0 for i in range(numHeadStates)}

    for state in states:
        if len(state) == 1:
            continue
        else:
            Belief[state] = Belief[state[:-1]] @ T[state[-1]]
            Z[state] = Z[state[:-1]] + \
                gamma**(len(state)-2)*(Belief[state[:-1]] @ C[state[-1]])

    result = {}
    for state in states:
        if (len(state) == window_length+2):
            result[state] = {'Z': Z[state], 'Belief': Belief[state]}
    return result


def thm1(sensingcost, result, actions, window_length, C, T, gamma, numHeadStates, V_head):
    lst = np.zeros(numHeadStates)
    for HeadState in range(numHeadStates):
        possibleKeys = [i for i in result.keys() if i[0] == HeadState]
        head_val = np.inf
        for key in possibleKeys:
            min_value = np.inf
            Belief = result[key]['Belief']
            # print(f'Belief: {Belief}')
            for action in actions:

                value = Belief@C[action] + gamma * \
                    (sensingcost + (Belief@T[action])@V_head.T)
                min_value = min(min_value, value)

            val = result[key]['Z'] + gamma**(window_length+1)*min_value
            head_val = min(head_val, val)

        lst[HeadState] = head_val

    print(f'LHS_min: {lst}, Constrained Value Fn: {V_head}')
    # string_val = ['Stop', 'Continue']
    return lst >= V_head


if __name__ == "__main__":
    # actions = ["0", "1", "2", "3"]
    actions = ["R", "B"]
    numHeadStates = 2
    gamma = 0.5
    windowLength = 3
    sensingcost = 0.125

    T = {
        "R": np.array([[0.7, 0.3], [0.2, 0.8]]),
        "B": np.array([[0.3, 0.7], [0.9, 0.1]])
    }

    C = {
        "R": np.array([0.25, 0.75]),
        "B": np.array([1, 0.5])
    }

    # T = {
    #     "R": np.array([[0.28,0.72],[0.934,0.066]]),
    #     "B": np.array([[0.31,0.69],[0.481,0.519]])
    # }

    # C = {
    #     "R": np.array([0.066,0.502]),
    #     "B": np.array([0.29,0.41])
    # }
    sensingActions = [action + "S" for action in actions]
    states = generate_states(numHeadStates, actions, windowLength)
    print(f'Sensing Cost: {sensingcost}, Window_len: {windowLength}')
    V_head = np.zeros(numHeadStates)
    V_opt = np.zeros(numHeadStates)
    mdp = generate_pomdp(windowLength, T, deepcopy(
        C), gamma, actions, numHeadStates, sensingcost/gamma)
    # policy, value_function  = brute_force_search(states, actions+sensingActions, mdp, 0.9, windowLength)

    opt_policy, opt_val = brute_force_search(
        states, actions+sensingActions, mdp, gamma, windowLength)
    policy = {i: tuple([opt_policy[tuple([i])]]) for i in range(numHeadStates)}

    for i in range(numHeadStates):
        V_head[i] = opt_val[tuple([i])]

    result = Z(windowLength, T, deepcopy(C), numHeadStates, gamma)

    print(thm1(sensingcost/gamma, result, actions, windowLength,
          deepcopy(C), T, gamma, numHeadStates, V_head))
