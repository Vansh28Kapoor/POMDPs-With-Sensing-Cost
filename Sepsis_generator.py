import pickle
from new_planner import valueEvaluation, Q_pi, brute_force_search
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
import json
from new_Heuristic import Improved_Heuristic, Heuristic, Value_policy
import numpy as np
from copy import deepcopy
import gymnasium
import sys
import os
import pandas as pd
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
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
initial_state_df = initial_state_df = pd.read_csv('initialStateDistribution.csv', header=None)
initial_state_array = initial_state_df.to_numpy().reshape(-1)
ter = {715: 0.0}

actions = [str(i) for i in range(num_actions)]
sensingActions = [action + "S" for action in actions]
gamma = 0.99
T_dic = {actions[i]: T[i] for i in range(len(actions))}
C_dic = {actions[i]: C[:, i] for i in range(len(actions))}
np.savez('Sepsis_params.npz', C=C, T=T, init=initial_state_array)
zero_mdp = generate_pomdp(0, deepcopy(T_dic), deepcopy(
    C_dic), gamma, actions, num_states, 0)

opt_policy, opt_val = brute_force_search([tuple([i]) for i in range(
    num_states)], actions+sensingActions, zero_mdp, gamma, 0, ter)

policy = []
value = []
for i in range(num_states):
    policy.append(opt_policy[tuple([i])][:-1])
    value.append(opt_val[tuple([i])])
with open("Sepsis_new.pkl", 'wb') as f:
    pickle.dump((value, policy), f)