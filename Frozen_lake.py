import numpy as np
#import pulp
import math
import argparse
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
from new_planner import valueEvaluation, Q_pi, brute_force_search
from Heuristic import Solve, V_blind
import gymnasium
from copy import deepcopy
import pickle



actions = ["0", "1", "2", "3"]
sensingActions = [action + "S" for action in actions]
numHeadStates = 16
windowLength = 0
alpha = 0.9
gamma = alpha
sensingcost = 0  #Add True sensing cost without gamma factor 

## Environment Dynamics Calc

## For Modified Maps
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
map = maps["4x4"]
map_name = "4x4"
env = gymnasium.make('FrozenLake-v1', desc=map,
                         map_name=map_name, is_slippery=True)

## For Default Maps
# default_maps = {
#     "4x4":[
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFG"
#     ],

# "8x8": [
#     "SFFFFFFF",
#     "FFFFFFFF",
#     "FFFHFFFF",
#     "FFFFFHFF",
#     "FFFHFFFF",
#     "FHHFFFHF",
#     "FHFFHFHF",
#     "FFFHFFFG",
# ],
# }
# env = gymnasium.make('FrozenLake-v1', desc=default_maps["4x4"], map_name="4x4", is_slippery=True)


goal = []
terminal_states = set()
P_dic = env.unwrapped.P
num_actions = env.action_space.n
num_states = env.observation_space.n
T = {i: np.zeros((num_states, num_states)) for i in actions}
C = {i: np.zeros(num_states) for i in actions}
for state in P_dic:
    for action in P_dic[state]:
        for probability, next_state, reward, ter in P_dic[state][action]:
            T[actions[action]][state,next_state] += probability
            if ter:
                terminal_states.add(next_state)
            if reward>0:
                C[actions[action]][state] -= reward*probability
            if not goal and reward>0:
                goal.append(next_state)


ter = {terminal:0.0 for terminal in terminal_states}

zero_mdp = generate_pomdp(0, deepcopy(T), deepcopy(C), gamma, actions, num_states, 0)

opt_policy, opt_val = brute_force_search([tuple([i]) for i in range(num_states)], actions+sensingActions, zero_mdp, gamma, 0, ter)

policy = []
value = []
for i in range(num_states):
    policy.append(opt_policy[tuple([i])][0])
    value.append(opt_val[tuple([i])])
with open("Frozen_lake_custom_4.pkl", 'wb') as f:
    pickle.dump((value,policy),f)