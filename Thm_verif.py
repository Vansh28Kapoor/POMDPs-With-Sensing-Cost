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


def delta(result, actions, window_length, C, T, V_opt, gamma, numHeadStates, V_head):
    lst = np.zeros(numHeadStates)
    for HeadState in range(numHeadStates):
        possibleKeys = [i for i in result.keys() if i[0] == HeadState]
        head_val = np.inf
        for key in possibleKeys:
            Belief = result[key]['Belief']
            # print(f'Belief: {Belief}')
            val = result[key]['Z'] + gamma**(window_length+1)*V_blnd(Belief, actions, C, T, V_opt, gamma)
            head_val = min(head_val, val)

        lst[HeadState] = head_val
    
    # lets now try to evaluate the 2nd minimization term
    rel_diff = np.array(V_head) - np.array(lst) #diff between V_head (constrained Value fn.) - LHS_min
    max_diff = [max(0,np.max(rel_diff))]*len(V_head) # list containing max_difference+
    arg_max = np.argmax(rel_diff)
    mask = np.ones(len(V_head), dtype= bool)
    mask[arg_max] = False
    max_2 = max(0,np.max(rel_diff[mask])) #2nd maximum value
    max_diff[arg_max] = max_2
    LHS_min_2 = np.array(V_head) - (gamma*np.array(max_diff))
    # print(LHS_min_2, np.array(lst))
    LHS_min_final = np.minimum(np.array(lst) ,LHS_min_2)



    # print(f'LHS_min_Final: {LHS_min_final}, V_opt: {V_opt}, Constrained Value Fn: {V_head}, LHS_min: {lst}, LHS_min_2: {LHS_min_2}' )

    return V_head - LHS_min_final


def V_blnd(B, actions, C, T, V_opt, gamma):
    num_actions = len(C.keys())
    num_HeadStates = len(list(C.values())[0])
    Q = np.zeros(
        num_actions*num_HeadStates).reshape(num_HeadStates, num_actions)
    for i in range(num_actions):  # no. of actions
        Q[:, i] = T[actions[i]]@V_opt.T
        Q[:, i] = C[actions[i]] + gamma*Q[:, i]
    # Q = C + gamma * Q
    Final = B@Q
    return np.min(Final)

# C is s x a, T[0] is T for action 0, V is optimal


def thr_verif(C, V, gamma):
    Q = np.zeros((C[0].shape[0], len(C)))
    for i in range(T.shape[0]):  # no. of actions
        Q[:, i] = T[i]@V.T
    Q = C + gamma * Q
    minin = np.inf
    for action1 in range(Q.shape[1]):
        for action2 in range(Q.shape[1]):
            minin = min(np.min(T[action1]@(Q(action2).T-V.T)), minin)
    return minin


if __name__ == "__main__":
    actions = ["0", "1", "2", "3"]
    # actions = ["R", "B"]
    sensingActions = [action + "S" for action in actions]
    numHeadStates = 4
    windowLength = 1
    sensingcost = 0.064 # Sufficient to change sensing cost only here
    gamma = 0.8
    states = generate_states(numHeadStates, actions, windowLength)
    print(f'Sensing Cost: {sensingcost}, Win_len: {windowLength}')

    ## For Illustration 1
    # T = {
    #     "R": np.array([[0.7, 0.3], [0.2, 0.8]]),
    #     "B": np.array([[0.3, 0.7], [0.9, 0.1]])
    # }

    # C = {
    #     "R": np.array([0, 1]),
    #     "B": np.array([1, 0])
    # }
    
    ## For Case Study 
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

    ## For Counter Example
    # T = {
    #     "R": np.array([[0.28,0.72],[0.934,0.066]]),
    #     "B": np.array([[0.31,0.69],[0.481,0.519]])
    # }

    # C = {
    #     "R": np.array([0.066,0.502]),
    #     "B": np.array([0.29,0.41])
    # }

    V_head = np.zeros(numHeadStates)
    V_opt = np.zeros(numHeadStates)
    mdp = generate_pomdp(windowLength, T, deepcopy(
        C), gamma, actions, numHeadStates, sensingcost/gamma)
    zero_mdp = generate_pomdp(0, T, deepcopy(
        C), gamma, actions, numHeadStates, 0)
    # policy, value_function  = brute_force_search(states, actions+sensingActions, mdp, 0.9, windowLength)

    opt_policy, opt_val = brute_force_search(
        states, actions+sensingActions, mdp, gamma, windowLength)
    _, zero_val = brute_force_search([tuple([i]) for i in range(
        numHeadStates)], actions+sensingActions, zero_mdp, gamma, 0)
    policy = {i: tuple([opt_policy[tuple([i])]]) for i in range(numHeadStates)}

    for i in range(numHeadStates):
        V_head[i] = opt_val[tuple([i])]
        V_opt[i] = zero_val[tuple([i])]

    result = Z(windowLength, T, deepcopy(C), numHeadStates, gamma)
    # print(V_head)
    print(f"Suboptimality Gap: {delta(result, actions, windowLength, deepcopy(
        C), T, V_opt, gamma, numHeadStates, V_head)}")

    # for i in range(numHeadStates):
    #     while (policy[i][-1][-1] != 'S'):
    #         policy[i] = policy[i] + tuple([opt_policy[tuple([i])+policy[i]]])

    # for i in range(numHeadStates):
    #     print(f"state {i}, policy: {policy[i]}, value: {opt_val[tuple([i])]}")
