import numpy as np
#import pulp
import math
import argparse
from generate_mdp import generate_states


def valueEvaluation(policy, states, actions, transition, gamma, window_len):
    v0 = {state: 0 for state in states}
    v1 = {state: 0 for state in states}
    max_error = 1e-10
    count = 0

    a0 = 0
    while(1):
        a0+=1
        error = -1
        for state in states:
            v1[state] = 0
            for nextState in transition[state][policy[state]].keys():
                prob = transition[state][policy[state]][nextState]['prob']
                cost = transition[state][policy[state]][nextState]['cost']
                v1[state] += (prob*(cost + (gamma*v0[nextState])))
            error = max(error, np.abs(v1[state]-v0[state]))
        vp0 = {k:round(v, 3) for k,v in v0.items()}
        vp1 = {k:round(v, 3) for k,v in v1.items()}
        if error < max_error:
            break

        v0 = v1.copy()
        count += 1

    return v1

def Q_pi(V, s, a, transition, gamma):
    return sum([transition[s][a][s_]['prob']*(transition[s][a][s_]['cost']+gamma*V[s_]) for s_ in transition[s][a].keys()])

def brute_force_search(states, actions, transition, gamma, window_len):
    policy = {state: list(transition[state].keys())[0] for state in states}
    while(True):
        V = valueEvaluation(policy, states, actions, transition, gamma, window_len)
        improved_policy = policy.copy()
        improvable = False
        for s in states:
            for a in actions:
                if a == improved_policy[s] or a not in transition[s].keys():
                    continue
                t_policy = policy.copy()
                if(V[s] - Q_pi(V, s, a, transition, gamma) > 1e-10):
                    improved_policy[s] = a
                    improvable = True
                    break
        if not improvable:
            break
        else:
            policy = improved_policy.copy()
        # print(policy)
    # print("out of while")
    value_function = valueEvaluation(policy, states, actions, transition, gamma, window_len)
    policy = policy
    return policy, value_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp")
    parser.add_argument("--policy",default= "")
    parser.add_argument("--optimal", action=argparse.BooleanOptionalAction)
    parser.add_argument("--window_len", default = -1)
    parser.add_argument("--print_all", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    file = args.mdp
    pol_file = args.policy
    optimal = args.optimal
    window_len  = int(args.window_len)

    with open(file) as f:
        content = f.readlines()
    file = [content[i][:-1].split() for i in range(len(content))]   

    mdp ={}
    mdp['numStates'] = int(file[0][1])
    mdp['numActions'] = int(file[1][1])
    actions = ['R', 'B', 'RS', 'BS']
    states0, states1 = generate_states(WINDOW_LEN = window_len)
    states = states0 + states1
    mdp['end'] = []
    for i in range(1,len(file[2])):
        mdp['end'].append(int(file[2][i]))

    mdp['transition'] = {state: {} for state in states} #[[[] for i in range(mdp['numActions'])] for j in range(mdp['numStates'])]
    for line in file:
        if(line[0] == 'transition'):
            state = line[1]
            action = line[2]
            nextState = line[3]
            if state in states and nextState in states:
                if action not in mdp['transition'][state].keys():
                    mdp['transition'][state][action] = {}
                mdp['transition'][state][action][nextState] = {'cost': float(line[4]), 'prob':float(line[5])}
    print(mdp['transition'])
    mdp['discount'] = float(file[-1][1])


    numStates  = mdp['numStates']
    numActions = mdp['numActions']
    transition = mdp['transition']
    gamma      = mdp['discount']
    # print(transition)
    end_states = mdp['end']


    if pol_file != "":
        given_policy = {state:"RS" for state in states}
        with open(pol_file) as p:
            pol_content = p.readlines()
        pol_file = [pol_content[i].strip() for i in range(len(pol_content))]   
        for line in pol_content:
            if line == "":
                continue
            state = line.split(" ")[0]
            action = line.split(" ")[1].replace("\n", "")
            action = state + action
            for i in range(len(action)):
                if not action[i+1] == 'S':
                    given_policy[action[:i+1]] = action[i+1]
                else:
                    given_policy[action[:i+1]] = action[i:]
                    break
        V0 = valueEvaluation(given_policy, states, actions, transition, gamma, window_len)
        print(V0['0'], V0['1'])

    if(optimal):
        opt_policy, opt_val = brute_force_search(states, actions, transition, gamma, window_len)
        policy = {0:opt_policy['0'], 1:opt_policy['1']}
        while(policy[0][-1] != 'S'):
            policy[0] = policy[0] + opt_policy['0'+policy[0]]
        while(policy[1][-1] != 'S'):
            policy[1] = policy[1] + opt_policy['1'+policy[1]]
        if args.print_all:
            for s in states:
                print(f"state {s}, policy: {opt_policy[s]}, value: {opt_val[s]}")
        else:
            print(f"state 0, policy: {policy[0]}, value: {opt_val['0']}")
            print(f"state 1, policy: {policy[1]}, value: {opt_val['1']}")