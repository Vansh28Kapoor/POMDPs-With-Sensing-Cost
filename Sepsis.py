import numpy as np
from copy import deepcopy
import gymnasium
import pickle
import os
import json
from time import time as tme
import numpy as np
import pandas as pd

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
        action_state = []
        cum_val = 0
        time = 0
        belief = np.zeros(C.shape[0])
        belief[state] = 1

        act, value = V_blind(belief, C, T, V, gamma)
        # print(f'Belief: {belief}, V_blind: {V_blind(belief, C, T, V, gamma)}, Compare: {belief@V}')
        diff = value-(belief@V)

        while (diff < k and time <= max and state not in ter):
            action_state += [str(act)]
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


def Improved_Heuristic(pi, d, C, T, gamma, k, max=100, ter={}):
    V_old = Value_policy(pi, C, T, gamma, k, ter)
    start = tme()
    new_pi = Q_Heuristic(pi, d, C, T, V_old, gamma, k, max, ter)
    stop = tme()
    V_new = Value_policy(new_pi, C, T, gamma, k, ter)
    initial_state_df = pd.read_csv('initialStateDistribution.csv', header=None)
    initial_state_array = initial_state_df.to_numpy().reshape(-1)
    diff = (V_old-V_new)@initial_state_array
    iter = 0
    print('Iteration:', iter, 'Time:', stop-start, "New Value Function:", V_new@initial_state_array, 'Delta:', diff, flush=True)
    while diff > 1e-6:
        pi = new_pi
        V_old = V_new
        start = tme()
        new_pi = Q_Heuristic(pi, d, C, T, V_old, gamma, k, max, ter)
        stop = tme()
        V_new = Value_policy(new_pi, C, T, gamma, k, ter)
        diff = (V_old-V_new)@initial_state_array
        iter += 1
        print('Iteration:', iter, 'Time:', stop-start, "New Value Function:", V_new@initial_state_array, 'Delta:', diff, flush=True)
    return new_pi, V_new


# For Q_Heuristic => V is the Value function for root states for policy pi

# Here we are outputing an improved policy
def Q_Heuristic(pi, d, C, T, V, gamma, k, max=100, ter={}):
    new_pi = []
    for state in range(len(pi)):
        if state in ter:
            new_pi.append(pi[state])
            continue
        new_act = []
        B = np.zeros(len(pi))
        B[state] = 1.0
        cum_val = 0
        time = 0
        # restricting to actions taken from states in S_{d}, i.e., till layer d
        for action in pi[state][:d+1]:
            action = int(action)
            act_heuristic, val_heuristic = VQ_Heuristic(
                B, C, T, V, gamma, k, max)
            if val_heuristic < ((V[state]-cum_val)/(gamma**time))-1e-6: ## Equivalent way for checking l19 of Algo
                # Instead of comparing value function we simply compare Q-values! (faster and equivalent!)

                ## For checking consistency with l19
                # check_pi = deepcopy(pi)
                # check_pi[state] = act_heuristic
                # check_val = Value_policy(check_pi, C, T, gamma, k, ter)
                # print(state, act_heuristic, 'Check:', check_val[state], 'Value:', val_heuristic, 'Original:', V[state])
                
                new_act += act_heuristic
                new_pi.append(new_act)
                break
            new_act = new_act + [str(action)]
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
            if action >= num_actions:
                action = action - num_actions
            cum_val += (gamma**time)*(B@C[:, action])
            time += 1
            B = B@T[action]
        cum_val += k*(gamma**(time-1))
        val.append(cum_val)
        steps.append(time)
        bel.append(B)
    return Solve(val, bel, steps, gamma, ter)


def VQ_Heuristic(B, C, T, V, gamma, k, max=100):
    start = tme()
    act_state = []
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
        act_state += [str(act_blind)]
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
    act_state += [str(act_sense)]
    time += 1
    belief = belief@T[act_sense]
    cum_val += (belief@V.T)*(gamma**time)
    print("VQ_Heuristic Time:", tme()-start, flush=True)
    return act_state, cum_val

num_states = 716 
num_actions = 25 

transition_df = pd.read_csv('transitionFunction.csv', header=None)
transition_array = transition_df.to_numpy()  # Shape: (S * A, S)
T = np.zeros((num_actions, num_states, num_states))
for si in range(num_states):
    for a in range(num_actions):
        row_idx = si * num_actions + a
        T[a, si, :] = transition_array[row_idx, :]
C = np.zeros((num_states, num_actions))
for state in range(num_states):
    for action in range(num_actions):
        if state != 715:
            C[state, action] = -T[action, state, 714]
ter = {715: 0.0}
gamma = 0.99
k = 0.05

if __name__ == "__main__":
    initial_state_df = pd.read_csv('initialStateDistribution.csv', header=None)
    initial_state_array = initial_state_df.to_numpy().reshape(-1)
    with open("Sepsis_new.pkl", "rb") as f:
        opt_fun, opt_pol = pickle.load(f)
    for i in range(len(opt_pol)):
        if opt_pol[i][-1] == 'S':
            opt_pol[i] = [opt_pol[i][:-1]]
        else:
            opt_pol[i] = [opt_pol[i]]
    opt_fun = np.array(opt_fun)

    ## Testing ATM & SPI
    for k in [0.1, 0.05, 0.01, 0.005]:
        start = tme()
        value_fn, pi = Heuristic(deepcopy(C), deepcopy(
            T), deepcopy(opt_fun), gamma, k/gamma, max=500, ter=ter)
        stop = tme()
        print(f"Heuristic Time for k={k}:", stop-start, flush=True)
        print(f"Heuristic Value for k={k}:", value_fn@initial_state_array, flush=True)
        start = tme()
        V0 = Improved_Heuristic(opt_pol, 0, deepcopy(C), deepcopy(
            T), gamma, k, max=500, ter=ter)[1]
        stop = tme()
        print(f"SPI Time for k={k}:", stop-start, flush=True)
        print(f"SPI Value for k={k}:", V0@initial_state_array, flush=True)

        ## Testing SARSOP
        # file_path = os.path.expanduser(f"SARSOP/Sepsis_500_{k}.json")
        # with open(file_path, 'r') as f:
        #     policy = list(json.load(f))
        # SARSOP = Value_policy(policy, C, T, gamma, k, ter = ter)
        # print("Sensing Cost: ", k, "SARSOP Value", SARSOP@initial_state_array)