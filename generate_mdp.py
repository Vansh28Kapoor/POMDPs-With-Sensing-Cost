import numpy as np
import argparse
from utils.generate_mdp_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", default=0.1, type=float)
    parser.add_argument("--window_len", default=1, type=int)
    parser.add_argument("--seed", default=-1)
    parser.add_argument("--mdp_params", default=None)  # New argument for mdp_params file path

    args = parser.parse_args()
    K = args.K
    WINDOW_LEN = args.window_len
    gamma = 0.5

    if args.seed != -1:
        np.random.seed(int(args.seed))
    
    # Initialize MDP parameters from the file if seed is -1
    if args.seed == -1 and args.mdp_params:
        params = load_mdp_params_from_file(args.mdp_params)
        Cr = params['Cr']
        Cb = params['Cb']
        Tr = params['Tr']
        Tb = params['Tb']
    else:
        p0 = np.random.rand()
        p1 = np.random.rand()
        p2 = np.random.rand()
        p3 = np.random.rand()
        Tr = np.array([[p0, 1-p0], [p1, 1-p1]])
        Tb = np.array([[p2, 1-p2], [p3, 1-p3]])
        Cr = np.array([0, 1])
        Cb = np.array([1, 0])   
    
    Crs = Cr + gamma * K
    Cbs = Cb + gamma * K

    T = {}
    T['R'] = Tr
    T['B'] = Tb
    T['RS'] = Tr
    T['BS'] = Tb

    C = {}
    C['R'] = Cr
    C['B'] = Cb
    C['RS'] = Crs
    C['BS'] = Cbs

    print_pomdp(WINDOW_LEN, T, C, gamma)