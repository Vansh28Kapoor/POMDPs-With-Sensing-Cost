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
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

env = gymnasium.make('Taxi-v3')
p_slip = 0.1
# env =  gymnasium.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
terminal_states = set({500})
P_dic = env.unwrapped.P
num_actions = env.action_space.n
num_states = env.observation_space.n + 1
print(num_states, num_actions)
T = np.zeros((num_actions, num_states, num_states))
C = np.zeros((num_states, num_actions))
for state in P_dic:
    for action in P_dic[state]:
        if action < 4:
            row = state//100
            leftover = state % 100
            col = leftover//20
            leftover = leftover % 20
            if action == 0:
                true = 100*min(row+1, 4) + 20*col + leftover
                slip_1 = 100*row + 20*min(col+1, 4) + leftover
                slip_2 = 100*row + 20*max(col-1, 0) + leftover
            elif action == 1:
                true = 100*max(row-1, 0) + 20*col + leftover
                slip_1 = 100*row + 20*min(col+1, 4) + leftover
                slip_2 = 100*row + 20*max(col-1, 0) + leftover
            elif action == 2:
                true = 100*row + 20*min(col+1, 4) + leftover
                slip_1 = 100*max(row-1, 0) + 20*col + leftover
                slip_2 = 100*min(row+1, 4) + 20*col + leftover
            else:
                true = 100*row + 20*max(col-1, 0) + leftover
                slip_1 = 100*max(row-1, 0) + 20*col + leftover
                slip_2 = 100*min(row+1, 4) + 20*col + leftover
            T[action, state, true] = 1-2*p_slip
            T[action, state, slip_1] += p_slip
            T[action, state, slip_2] += p_slip
            C[state, action] += 1
            continue

        for probability, next_state, reward, ter in P_dic[state][action]:
            if reward == 20:
                T[action, state, 500] = 1
                C[state, action] = -20
                continue
            T[action, state, next_state] += probability
            C[state, action] -= reward*probability
ter = {terminal: 0.0 for terminal in terminal_states}

actions = ["0", "1", "2", "3", "4", "5"]
sensingActions = [action + "S" for action in actions]
gamma = 0.95
T_dic = {actions[i]: T[i] for i in range(len(actions))}
C_dic = {actions[i]: C[:, i] for i in range(len(actions))}
np.savez('Taxi_params.npz', C=C, T=T)
zero_mdp = generate_pomdp(0, deepcopy(T_dic), deepcopy(
    C_dic), gamma, actions, num_states, 0)

opt_policy, opt_val = brute_force_search([tuple([i]) for i in range(
    num_states)], actions+sensingActions, zero_mdp, gamma, 0, ter)

policy = []
value = []
for i in range(num_states):
    policy.append(opt_policy[tuple([i])][0])
    value.append(opt_val[tuple([i])])
with open("Taxi.pkl", 'wb') as f:
    pickle.dump((value, policy), f)
