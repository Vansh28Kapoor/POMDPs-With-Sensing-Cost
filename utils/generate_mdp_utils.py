import numpy as np
import argparse

def binary_to_string(num, len_action):
    binary_string = "{0:b}".format(num)
    binary_string = "0"*(len_action-len(binary_string))+binary_string
    string = ""
    for char in binary_string:
        if char == '0':
            string += 'R'
        else:
            string += 'B'
    return string

def generate_states(WINDOW_LEN):
    states0 = ['0']
    for w in range(1, WINDOW_LEN+1):
        for i in range(2**w):
            states0.append('0'+binary_to_string(i, w))
    states1 = ['1']
    for w in range(1, WINDOW_LEN+1):
        for i in range(2**w):
            states1.append('1'+binary_to_string(i, w))
    return states0, states1

def load_mdp_params_from_file(filepath):
    mdp_params = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:  # skip lines without the colon character
                continue
            key, values = line.split(':')
            mdp_params[key.strip()] = np.array(eval(values.strip()))
    return mdp_params

def print_pomdp(window_len, T, C, gamma):
    numStates = 2**(window_len + 2) - 2       
    numActions = 4
    # actions = R, B, RB, RR, BR, BB
    print("numStates", numStates)
    print("numActions", numActions)
    print("end -1")

    states0, states1 = generate_states(window_len)
    actions = ['R', 'B', 'RS', 'BS']
    mdp = {state: {action: {} for action in actions} for state in states0+states1}
    belief = {"0":np.array([1, 0]), "1":np.array([0, 1])}

    for state in [0,1]:
        for action in ['RS', "BS"]:
            for nextState in [0, 1]:
                mdp[str(state)][action][str(nextState)] = {}
                prob = T[action][state][nextState]
                cost = C[action][state]
                mdp[str(state)][action][str(nextState)]['prob'] = prob
                mdp[str(state)][action][str(nextState)]['cost'] = cost

    for state in states0[1:]+states1[1:]:
        belief[state] = belief[str(state[:-1])]@T[state[-1]]
        for action in ["RS", "BS"]:
            for nextState in [0, 1]:
                mdp[state][action][str(nextState)] = {}
                prob = (belief[state]@T[action])[nextState]
                cost = belief[state]@C[action]
                mdp[state][action][str(nextState)]['prob'] = prob
                mdp[state][action][str(nextState)]['cost'] = cost

    for state in states0+states1:
        if(len(state) <= window_len):
            for action in ["R", "B"]:
                nextState = state + action
                prob = 1
                cost = belief[state]@C[action]
                mdp[state][action][nextState] = {}
                mdp[state][action][nextState]['prob'] = prob
                mdp[state][action][nextState]['cost'] = cost


    for state in states0+states1:
        for action in actions:
            for nextState in mdp[state][action].keys():
                cost = mdp[state][action][nextState]['cost']
                prob = mdp[state][action][nextState]['prob']
                print("transition ", state, " ", action, " ", nextState, " ", cost, " ", prob)

    print("gamma ", gamma)
    # print belief dictionary into a text file
    with open("belief.txt", "w") as f:
        for key in belief.keys():
            f.write("%s %s  " % (key, belief[key][0]))
            f.write("\n")