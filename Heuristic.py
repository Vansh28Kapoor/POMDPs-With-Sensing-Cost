import numpy as np
from copy import deepcopy

# C is s x a, T[0] is T for action 0, V is optimal
def V_blind(B, C, T, V, gamma):
    Q = np.zeros(C.shape)
    for i in range(T.shape[0]):  # no. of actions
        Q[:, i] = T[i]@V.T
    Q = C + gamma * Q
    Final = B@Q
    return np.argmin(Final), np.min(Final)


def Heuristic(C, T, V, gamma, k, max=100):
    action = []
    val = []
    bel = []
    steps = []
    for state in range(C.shape[0]):
        print(state)
        action_state = ''
        cum_val = 0
        time = 0
        belief = np.zeros(C.shape[0])
        belief[state] = 1

        act, value = V_blind(belief, C, T, V, gamma)
        print(f'Belief: {belief}, V_blind: {V_blind(belief, C, T, V, gamma)}, Compare: {belief@V}')
        diff = value-(belief@V)

        while (diff < k and time <= max):
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

    return Solve(val, bel, steps, gamma), action


def Solve(val, bel, steps, gamma):
    bel = np.array(bel)
    val = np.array(val)
    steps = np.array(steps)
    I = np.eye(bel.shape[0])
    new = (gamma**steps)[:, np.newaxis]
    A = I - (new*bel)
    solution = np.linalg.solve(A, val)
    return solution


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


T = np.array([
    [[0.7, 0.3], [0.2, 0.8]],
    [[0.3, 0.7], [0.9, 0.1]]
    ])

C = np.array([
    [0.25, 0.75],
    [1, 0.5]]).T

V = np.array([0.56818182, 0.79545455])

if __name__ == "__main__": 
    print(Heuristic(deepcopy(C), deepcopy(T), deepcopy(V), 0.5, 0.5, max=500))
