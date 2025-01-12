import numpy as np
from copy import deepcopy
import gymnasium
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
from new_Heuristic import Improved_Heuristic, Heuristic, Value_policy
import json
import pickle
import time


gamma = 0.95
k = 1
ter = {500:0.0}
with open("Taxi.pkl",'rb') as f:
    V_taxi, policy_taxi = pickle.load(f)
V_taxi = np.array(V_taxi)
taxi_params = np.load('Taxi_params.npz')
C = taxi_params['C']
T = taxi_params['T']

start_time = time.perf_counter()
value_fn, pi = Heuristic(deepcopy(C), deepcopy(T), deepcopy(V_taxi), gamma, k/gamma, max=200, ter = ter)
end_time = time.perf_counter()
print(end_time-start_time)


start_time = time.perf_counter()
V0 = Improved_Heuristic(policy_taxi,0, deepcopy(C), deepcopy(T), np.array(deepcopy(Value_policy(policy_taxi, C, T, gamma, k, ter))), gamma, k, max=200, ter = ter)[1]
end_time = time.perf_counter()
print(end_time-start_time)


start_time = time.perf_counter()
file_path = os.path.expanduser("~/Downloads/SARSOP/SARSOP_taxi_1.json")
with open(file_path, 'r') as f:
    policy = list(json.load(f))
SARSOP = Value_policy(policy, C, T, gamma, k, ter = ter)
end_time = time.perf_counter()
print(end_time-start_time)


total_sum_SARSOP = 0
total_sum_heuristic = 0
total_sum_heuristic_old = 0
num = 0
for taxi_row in range(5):
    for taxi_col in range(5):
        for passenger_location in range(4):
            for destination in range(4):
                if passenger_location == destination:
                    continue
                index = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                total_sum_SARSOP += SARSOP[index]
                total_sum_heuristic += V0[index]
                total_sum_heuristic_old += value_fn[index]
                num += 1
avg_value_SARSOP = total_sum_SARSOP/num
avg_value_heuristic = total_sum_heuristic/num
total_sum_heuristic_old = total_sum_heuristic_old/num
print(total_sum_heuristic_old, avg_value_heuristic, avg_value_SARSOP)