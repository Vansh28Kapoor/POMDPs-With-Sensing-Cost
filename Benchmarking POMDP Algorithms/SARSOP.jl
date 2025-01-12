using PyCall
using POMDPModels
using LinearAlgebra
using POMDPs: solve
using POMDPs
using NativeSARSOP: SARSOPSolver
using POMDPTools  # For DiscreteBelief
using NativeSARSOP
import QuickPOMDPs: QuickPOMDP
import POMDPTools: ImplicitDistribution
import Distributions: Categorical
using JSON
using Pickle
maps = Dict(
    "4x4" => [
        "FHSF",
        "FGHF",
        "FHHF",
        "FFFF"
    ],
    "8x8" => [
        "FFFFFSFF",
        "FFFFFFFF",
        "HHHHHHFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FHFFFHHF",
        "FHFFHFHH",
        "FGFFFFFF"
    ]
)

default_maps = Dict(
        "4x4" => [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ],

    "8x8" => [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
)

map_name = "4x4"  # You can switch between "4x4" and "8x8"
custom_map = maps[map_name]
default_map = default_maps[map_name]

py"""
import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=$custom_map, map_name="$map_name", is_slippery=True).unwrapped  # Access the core environment
"""
P_dic = py"env.P"  # Access the transition probabilities
num_actions = py"env.action_space.n"
num_actions = convert(Int, num_actions)
num_states = py"env.observation_space.n"
num_states = convert(Int, num_states)
println(num_states)

T = zeros(Float64, (num_states, num_actions, num_states))
R = zeros(Float64, (num_states, num_actions))
goal = Int[]
terminal_states = Set{Int}()

for state in keys(P_dic)
    for action in keys(P_dic[state])
        for (probability, next_state, reward, ter) in P_dic[state][action]
            T[state+1, action+1, next_state+1] += probability
            if ter
                push!(terminal_states, next_state+1)
            end
            if reward>0
                R[state+1, action+1] += reward*probability*1000
            end
            if isempty(goal) && reward > 0
                push!(goal, next_state+1)
            end
        end
    end
end

gamma = 0.9
k = 50


### POMDP Formulation
Transition = zeros(num_states, num_actions*2, num_states) # |S| x |A| x |S'|, T[sp, a, s] = p(sp | a, s)
Transition[:,1:num_actions,:] = deepcopy(T)
Transition[:,num_actions+1:num_actions*2,:] = deepcopy(T)

O = zeros(num_states+1, num_actions*2, num_states) # |O| x |A| x |S'|, O[o, a, sp] = p(o | a, sp)
for a in 1:num_actions
     O[1:num_states, a, 1:num_states] = Matrix{Float64}(I, num_states, num_states) ## First num_actions are SENSE actions
 end
O[num_states+1,num_actions+1:num_actions*2, 1:num_states] .= 1.0  # Next num_actions are Blind actions
     
discount = 0.9
Reward = zeros(num_states, num_actions*2)
Reward[:,1:num_actions] = deepcopy(R)
Reward[:,num_actions+1:num_actions*2] = deepcopy(R)
Reward[:,1:num_actions, :] .-= k

tabular_pomdp = QuickPOMDP(
    states = 1:num_states,
    actions = 1:(num_actions*2),
    observations = 1:num_states+1,
    discount = gamma,


    transition = function (s, a)
        Categorical(Transition[s, a, :])
    end,

    observation = function (a, sp)
        Categorical(O[:, a, sp])
    end,

    reward = function (s, a)
        Reward[s, a]
    end,

    initialstate = Deterministic(3),
    isterminal = s -> s in terminal_states

)


##SARSOP
custom_lower = NativeSARSOP.BlindLowerBound(bel_res=1e-3)  # Modify parameters as needed
custom_upper = NativeSARSOP.FastInformedBound(bel_res=1e-3)  # Modify parameters as needed
start = time()
sarsop_solver = SARSOPSolver(; max_time= 1.0, precision=1e-3, delta=1e-1, epsilon = 0.5, kappa = 0.5, prunethresh = 0.1, init_lower = custom_lower, init_upper = custom_upper)
policy = solve(sarsop_solver, tabular_pomdp)

val_pol = Dict{Int, String}()
for i in 1:num_states
    b = zeros(num_states)
    b[i] = 1.0
    b = b'
    a = action(policy, b)
    str = ""
    iterr = 0
    while(a>4 && iterr<200)
        b = b*Transition[:,a,:]
        str *= string(a-5)
        a = action(policy, b)
        iterr += 1
    end
    str *= string(a-1)
    val_pol[i] = deepcopy(str)
end
end_time = time()
println(end_time - start)

list = []  # Initialize an empty array
for i in 1:num_states
    # Slice the first 200 characters of the string
    if !(i in terminal_states)
        push!(list, val_pol[i][1:min(end, 200)])  # Add sliced value to the list
    else
        push!(list, "")  # Add an empty string for terminal states
    end
end

open(expanduser("~/Downloads/SARSOP/SARSOP_customgrid4_05.json"), "w") do file
    JSON.print(file, list)
end

