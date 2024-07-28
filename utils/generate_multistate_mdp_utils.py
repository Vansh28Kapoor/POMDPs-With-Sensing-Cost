import numpy as np
import argparse


def to_base_n(x, n):
    """
    Converts a number x to base n
    :param x: number to convert
    :param n: base to convert to
    :return: list of integers (0 to n-1) representing x in base n
    """
    if x == 0:
        return [0]

    symbols = []
    while x > 0:
        symbols.append(x % n)
        x //= n

    return symbols[::-1]


def nary_to_string(num, len_action, actions):
    """
    Converts a number to a string of actions
    :param num: number to convert
    :param len_action: length of the string to return
    :param actions: list of action symbols
    """
    num_actions = len(actions)
    nary_string = to_base_n(num, num_actions)
    nary_string = [0]*(len_action-len(nary_string))+nary_string

    return [actions[i] for i in nary_string]


def generate_states(num_head_states, actions, WINDOW_LEN):
    """
    Generates a dictionary of states
    :param num__head_states: number of head states
    :param actions: list of action symbols
    :param WINDOW_LEN: length of the window
    :return: list of states
    """
    states = []
    num_actions = len(actions)
    for i in range(num_head_states):
        states += [tuple([i])]
        for w in range(1, WINDOW_LEN+1):
            for j in range(num_actions**w):
                states.append(tuple([i]+nary_to_string(j, w, actions)))

    return states


def load_mdp_params_from_file(filepath):
    mdp_params = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:  # skip lines without the colon character
                continue
            key, values = line.split(':')
            mdp_params[key.strip()] = np.array(eval(values.strip()))
    return mdp_params


def generate_pomdp(windowLen, T, C, gamma, actions, numHeadStates, K):
    numNonSensingActions = len(actions)
    SensingActions = [action + "S" for action in actions]
    for action in actions:
        T[action+'S'] = T[action]
        C[action+'S'] = C[action]

    states = generate_states(numHeadStates, actions, windowLen)
    mdp = {state: {} for state in states}
    # print(mdp)
    for state in states:
        if len(state) <= windowLen:
            mdp[state] = {action: {} for action in actions+SensingActions}
            for action in actions:
                mdp[state][action][state+tuple([action])] = {}
        else:
            mdp[state] = {action: {} for action in SensingActions}

    for state in states:
        for action in SensingActions:
            for i in range(numHeadStates):
                try:
                    mdp[state][action][tuple([i])] = {}
                except KeyError:
                    print(state, mdp[state])
                    exit()

    # print(mdp)
    # belief is a dict where the keys are [i] for i in range(numHeadStates) and the corresponding value is a one hot numpy array
    belief = {tuple([i]): np.array([1 if i == j else 0 for j in range(
        numHeadStates)]) for i in range(numHeadStates)}

    # construct the belief vector for all of the states in the MDP.
    for state in states:
        if len(state) == 1:
            pass
        else:
            # belief[state] = belief[state[:-1]] (matrix multiply) T[state[-1]]
            belief[state] = np.matmul(belief[state[:-1]], T[state[-1]])

    for state in states:
        for action in actions:
            if len(state) < windowLen+1:
                mdp[state][action][state +
                                   tuple(action)]["cost"] = np.dot(belief[state], C[action])
                mdp[state][action][state+tuple(action)]["prob"] = 1

    for state in states:
        for action in SensingActions:
            for HeadStates in range(numHeadStates):
                mdp[state][action][tuple([HeadStates])]["cost"] = np.dot(
                    belief[state], C[action])+gamma*K
                mdp[state][action][tuple([HeadStates])]["prob"] = np.matmul(
                    belief[state], T[action])[HeadStates]
    return mdp

    # for state in states:
    #     for action in SensingActions:
    #         mdp[state][action][tuple([0])]["cost"] = np.dot(belief[state], C[action])+gamma*K
    #         mdp[state][action][tuple([1])]["cost"] = np.dot(belief[state], C[action])+gamma*K
    #         mdp[state][action][tuple([2])]["cost"] = np.dot(belief[state], C[action])+gamma*K
    #         # mdp[state][action][tuple([3])]["cost"] = np.dot(belief[state], C[action])+gamma*K

    #         mdp[state][action][tuple([0])]["prob"] = np.matmul(belief[state], T[action])[0]
    #         mdp[state][action][tuple([1])]["prob"] = np.matmul(belief[state], T[action])[1]
    #         mdp[state][action][tuple([2])]["prob"] = np.matmul(belief[state], T[action])[2]
    #         # mdp[state][action][tuple([3])]["prob"] = np.matmul(belief[state], T[action])[3]

    # # T is a 4x4 matrix. Access the first column of

    # return mdp
