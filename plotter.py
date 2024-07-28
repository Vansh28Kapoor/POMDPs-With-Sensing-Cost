import random,argparse,sys,subprocess,os
from generate_mdp import generate_states
import matplotlib.pyplot as plt
import numpy as np

# for W in range(0, 7):
W = 6
cmd_encoder = "python", "generate_mdp.py", "--K", str(10), "--window_len", str(W), "--seed", str(15)
f = open('mdp_file','w')
subprocess.call(cmd_encoder,stdout=f)
f.close()


cmd_encoder = "python", "planner.py","--mdp", "mdp_file", "--optimal", "--window_len", str(W)
output = subprocess.check_output(cmd_encoder,universal_newlines=True)
states0, states1 = generate_states(WINDOW_LEN = W)
states = states0 + states1

# open belief.txt and read number on each line
belief = {}
with open("belief.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]
for line in content:
    if line == "":
        continue
    state = line.split(" ")[0]
    prob = line.split(" ")[1]
    belief[state] = float(prob)

policy = {}
value = {}
for line in output.split("\n"):
    if line == "":
        continue
    state = line.split(" ")[0]
    policy[state] = line.split(" ")[1]
    value[state] = float(line.split(" ")[2])


# plot value of each state against belief of each state in a scatter plot with same actions having same color 

colors = ['r', 'b', 'g', 'y']
actions = ['R', 'B', 'RS', 'BS']
for state in states:
    if state in policy.keys():
        # if len(state)>=W+1:
        #     continue
        belief[state]
        value[state]
        plt.scatter(belief[state], value[state], s = 5, color=colors[actions.index(policy[state])])
        # plt.legend(loc='upper left')
plt.show()