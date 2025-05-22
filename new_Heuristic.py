import numpy as np
from copy import deepcopy
import gymnasium
import pickle
import os
import json
from time import time

# C is s x a, T[0] is T for action 0, V is optimal


def V_blind(B, C, T, V, gamma):
    Q = np.zeros(C.shape)
    for i in range(T.shape[0]):  # no. of actions
        Q[:, i] = T[i]@V.T
    Q = C + gamma * Q
    Final = B@Q
    return np.argmin(Final), np.min(Final)


def Q_val(C, T, V, gamma):
    Q = np.zeros(C.shape)
    for i in range(T.shape[0]):  # no. of actions
        Q[:, i] = T[i]@V.T
    Q = C + gamma * Q
    return Q


def V_blind_Q(B, Q):
    Final = B@Q
    return np.argmin(Final), np.min(Final)


# ATM Heuristic
def Heuristic(C, T, V, gamma, k, max=100, ter={}):
    action = []
    val = []
    bel = []
    steps = []
    for state in range(C.shape[0]):
        # print(state)
        action_state = ''
        cum_val = 0
        time = 0
        belief = np.zeros(C.shape[0])
        belief[state] = 1

        act, value = V_blind(belief, C, T, V, gamma)
        # print(f'Belief: {belief}, V_blind: {V_blind(belief, C, T, V, gamma)}, Compare: {belief@V}')
        diff = value-(belief@V)

        while (diff < k and time <= max and state not in ter):
            action_state += str(act)
            cum_val += (belief@C[:, act])*(gamma**time)
            time += 1
            belief = belief@T[act]

            act, value = V_blind(belief, C, T, V, gamma)

            diff = value-(belief@V)
        # print(cum_val)
        cum_val += k*(gamma**time)
        action.append(action_state)
        val.append(cum_val)
        steps.append(time)
        bel.append(belief)

    return Solve(val, bel, steps, gamma, ter), action


def Solve(val, bel, steps, gamma, ter={}):
    bel = np.array(bel)
    val = np.array(val)
    steps = np.array(steps)
    I = np.eye(bel.shape[0])
    new = (gamma**steps)[:, np.newaxis]
    A = I - (new*bel)
    for i in ter:
        A[i] = np.eye(A.shape[0])[i]
        val[i] = ter[i]
    solution = np.linalg.solve(A, val)
    return solution

# Our Improved Heuristic


def Improved_Heuristic(pi, d, C, T, V, gamma, k, max=100, ter={}):
    V_old = V
    new_pi = Q_Heuristic(pi, d, C, T, V_old, gamma, k, max, ter)
    V_new = Value_policy(new_pi, C, T, gamma, k, ter)
    diff = np.max(np.abs(V_old-V_new))
    while diff > 1e-6:
        pi = new_pi
        V_old = V_new
        new_pi = Q_Heuristic(pi, d, C, T, V_old, gamma, k, max, ter)
        V_new = Value_policy(new_pi, C, T, gamma, k, ter)
        diff = np.max(np.abs(V_old-V_new))
    return new_pi, V_new


# For Q_Heuristic => V is the Value function for root states for policy pi

# Here we are outputing an improved policy
def Q_Heuristic(pi, d, C, T, V, gamma, k, max=100, ter={}):
    new_pi = []
    for state in range(len(pi)):
        if state in ter:
            new_pi.append(pi[state])
            continue
        new_act = ''
        B = np.zeros(len(pi))
        B[state] = 1.0
        cum_val = 0
        time = 0
        # restricting to actions taken from states in S_{d}, i.e., till layer d
        for action in pi[state][:d+1]:
            action = int(action)
            act_heuristic, val_heuristic = VQ_Heuristic(
                B, C, T, V, gamma, k, max)
            # print(str(action), val_heuristic, (V[state]-cum_val)/(gamma**time))
            if val_heuristic < (V[state]-cum_val)/(gamma**time):
                new_act += act_heuristic
                new_pi.append(new_act)
                break
            new_act = new_act + str(action)
            cum_val += (B@C[:, action])*(gamma**time)
            B = B@T[action]
            time += 1
            if time == len(pi[state][:d+1]):
                new_act += pi[state][d+1:]
                new_pi.append(new_act)
    return new_pi


def Value_policy(pi, C, T, gamma, k, ter={}):
    val = []
    steps = []
    bel = []
    for state in range(len(pi)):
        cum_val = 0
        B = np.zeros(len(pi))
        B[state] = 1
        time = 0
        if state in ter:
            val.append(cum_val)
            steps.append(time)
            bel.append(B)
            continue

        for action in pi[state]:
            action = int(action)
            cum_val += (gamma**time)*(B@C[:, action])
            time += 1
            B = B@T[action]
        cum_val += k*(gamma**(time-1))
        val.append(cum_val)
        steps.append(time)
        bel.append(B)

    return Solve(val, bel, steps, gamma, ter)


def VQ_Heuristic(B, C, T, V, gamma, k, max=100):
    act_state = ''
    cum_val = 0
    time = 0
    belief = B
    Q = Q_val(C, T, V, gamma)
    act_sense, value_sense = V_blind(belief, C, T, V, gamma)
    value_sense += k
    Value_blind = belief@C + gamma * \
        np.array([V_blind_Q(belief@T[a], Q)[1] for a in range(T.shape[0])])
    act_blind, value_blind = np.argmin(Value_blind), np.min(Value_blind)
    value_blind += gamma*k
    while (value_blind < value_sense and time <= max):
        act_state += str(act_blind)
        cum_val += (belief@C[:, act_blind])*(gamma**time)
        time += 1
        belief = belief@T[act_blind]
        act_sense, value_sense = V_blind(belief, C, T, V, gamma)
        value_sense += k
        Value_blind = belief@C + gamma * \
            np.array([V_blind_Q(belief@T[a], Q)[1] for a in range(T.shape[0])])
        act_blind, value_blind = np.argmin(Value_blind), np.min(Value_blind)
        value_blind += gamma*k

    cum_val += (belief@C[:, act_sense]+k)*(gamma**time)
    act_state += str(act_sense)
    time += 1
    belief = belief@T[act_sense]
    cum_val += (belief@V.T)*(gamma**time)
    return act_state, cum_val

# Inventory Management
# T = np.array([[[1., 0., 0., 0.],
#              [1., 0., 0., 0.],
#     [0.5, 0.5, 0., 0.],
#     [0., 0.5, 0.5, 0.]],

#     [[1., 0., 0., 0.],
#      [0.5, 0.5, 0., 0.],
#      [0., 0.5, 0.5, 0.],
#      [0., 0., 0.5, 0.5]],

#     [[0.5, 0.5, 0., 0.],
#      [0., 0.5, 0.5, 0.],
#      [0., 0., 0.5, 0.5],
#      [0., 0., 0., 1.]],

#     [[0., 0.5, 0.5, 0.],
#      [0., 0., 0.5, 0.5],
#      [0., 0., 0., 1.],
#      [0., 0., 0., 1.]]

# ])


# C = np.array([[0,   -2.,   -2.75, -2.25],
#               [-1,   -1.75, -1.25, -0.75],
#               [-0.75, -0.25,  0.25,  0.75],
#               [0.75,  1.25,  1.75,  2.25]]).T

# V = np.array([-5.749999999666253,
#               -6.749999999666253,
#               -7.749999999666253,
#               -8.049999999666253])

# gamma = 0.8
# k = 0.2


# For Illustration 1
# T = np.array([
#     [[0.7, 0.3], [0.2, 0.8]],
#     [[0.3, 0.7], [0.9, 0.1]]
#     ])

# C = np.array([
#     [0, 1],
#     [1, 0]]).T

# V = np.array([0, 0])


# Frozen Lake
maps = {
    "4x4": [
        "FHSF",
        "FGHF",
        "FHHF",
        "FFFF"
    ],
    "8x8": [
        "FFFFFSFF",
        "FFFFFFFF",
        "HHHHHHFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FHFFFHHF",
        "FHFFHFHH",
        "FGFFFFFF"
    ],
}

default_maps = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

# For New_Maps
map = maps["4x4"]
map_name = "4x4"
env = gymnasium.make('FrozenLake-v1', desc=map,
                     map_name=map_name, is_slippery=True)

# For Default Maps
# map = default_maps["4x4"]
# map_name = "4x4"
# env = gymnasium.make('FrozenLake-v1', desc=map,
#                              map_name=map_name, is_slippery=True)

goal = []
terminal_states = set()
P_dic = env.unwrapped.P
num_actions = env.action_space.n
num_states = env.observation_space.n
T = np.zeros((num_actions, num_states, num_states))
C = np.zeros((num_states, num_actions))
for state in P_dic:
    for action in P_dic[state]:
        for probability, next_state, reward, ter in P_dic[state][action]:
            T[action, state, next_state] += probability
            if ter:
                terminal_states.add(next_state)
            if reward > 0:
                C[state, action] -= reward*probability
            if not goal and reward > 0:
                goal.append(next_state)


ter = {terminal: 0.0 for terminal in terminal_states}

PO_UCT_state16 = [
    "0",
    "00000000000000000000",
    "2222222111111",
    "22222111111",
    "3",
    "00000000000000000000",
    "00000000000000000000",
    "22111111",
    "0",
    "00000000000000000000",
    "00000000000000000000",
    "211111",
    "00",
    "1",
    "11",
    "1110000333313332130213313213131213012233"
]
gamma = 0.9
k = 0.005

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Benchmarking POMDP Algorithms", "FIB", "FIB_customgrid4_005.json")
    with open(file_path, 'r') as f:
        policy = list(json.load(f))
    start = time()
    # 1000 iteration, tolerence = 1e-3
    FIB = Value_policy(policy, C, T, gamma, k, ter=ter)
    stop = time()
    print(stop-start)
    print(FIB)

    # Evaluating SARSOP
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Benchmarking POMDP Algorithms", "SARSOP", "SARSOP_customgrid4_005.json")
    with open(file_path, 'r') as f:
        policy = list(json.load(f))
    start = time()
    SARSOP = Value_policy(policy, C, T, gamma, k, ter=ter)  # default
    stop = time()
    print(stop-start)
    print(f"SARSOP: {SARSOP[2]}")

    # Evaluating pi_atm Heuristic
    with open("Frozen_lake_custom_4.pkl", "rb") as f:
        opt_fun, opt_pol = pickle.load(f)
    opt_fun = np.array(opt_fun)
    start = time()
    value_fn, pi = Heuristic(deepcopy(C), deepcopy(
        T), deepcopy(opt_fun), gamma, k/gamma, max=200, ter=ter)
    stop = time()
    print(stop-start)

    # Evaluating our Improved Heuristic
    # "V" argument for the Improved_Heuristic should be the value function corresponding to "pi"
    start = time()
    V0 = Improved_Heuristic(opt_pol, 0, deepcopy(C), deepcopy(
        T), deepcopy(opt_fun), gamma, k, max=200, ter=ter)[1]
    stop = time()
    print(stop-start)
    print(V0[2])
