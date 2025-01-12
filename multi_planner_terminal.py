import numpy as np
# import pulp
import math
import argparse
from utils.generate_multistate_mdp_utils import generate_pomdp, generate_states
from new_planner import valueEvaluation, Q_pi, brute_force_search
from Heuristic import Solve, V_blind
import gymnasium
import time

if __name__ == "__main__":
    actions = ["L", "D", "R", "U"]
    sensingActions = [action + "S" for action in actions]
    numHeadStates = 64
    windowLength = 3
    alpha = 0.9
    sensingcost = 5e-2  # Add True sensing cost without gamma factor

    # Environment Dynamics Calc

    # For Modified/Customized Hard Maps
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

    # For Default Maps
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
    env = gymnasium.make(
        'FrozenLake-v1', desc=default_maps["8x8"], map_name="8x8", is_slippery=True)

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
                T[actions[action]][state, next_state] += probability
                if ter:
                    terminal_states.add(next_state)
                if reward > 0:
                    C[actions[action]][state] -= reward*probability
                if not goal and reward > 0:
                    goal.append(next_state)
    # for action in actions:
    #     for s in terminal_states:
    #         if s in goal:
    #             C[action][s] = -0.1
    #         else:
    #             C[action][s] = -0.00178

    ter = {terminal: 0.0 for terminal in terminal_states}

    # states contains all paths in the tree
    states = generate_states(numHeadStates, actions, windowLength)
    mdp = generate_pomdp(windowLength, T, C, alpha, actions,
                         numHeadStates, sensingcost/alpha)
    # MDP Format: {state: {action: {next_state:{'cost' & 'prob'} for all next_states }  for all actions }  for all states}

    start = time.perf_counter()
    opt_policy, opt_val = brute_force_search(
        states, actions+sensingActions, mdp, alpha, windowLength, ter)
    stop = time.perf_counter()
    print(stop-start)
    policy = {i: tuple([opt_policy[tuple([i])]]) for i in range(numHeadStates)}
    with open('opt_policy.txt', 'w') as f:
        f.write(str(opt_policy))
    for i in range(numHeadStates):
        if i in ter:
            continue
        while (policy[i][-1][-1] != 'S'):
            policy[i] = policy[i] + tuple([opt_policy[tuple([i])+policy[i]]])
    # Be careful about the 1/gamma & rounding-off
    lst = []
    for i in range(numHeadStates):
        # print(f"state {i}, policy: {policy[i]}, value: {opt_val[tuple([i])]}")
        lst.append(opt_val[tuple([i])])
    print(lst[0])
