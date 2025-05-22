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
using NPZ
import Random

valid_indices = []
for taxi_row in 0:4
    for taxi_col in 0:4
        for passenger_location in 0:3
            for destination in 0:3
                if passenger_location == destination
                    continue
                end
                index = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
                push!(valid_indices, index+1)
            end
        end
    end
end


file_path = joinpath(homedir(), "Downloads", "POMDP-MASTER", "Taxi", "Taxi_params.npz")
taxi_params = npzread(file_path)
C = taxi_params["C"]
R = C .* -1
T = taxi_params["T"]
T = permutedims(T, (2, 1, 3))
num_actions = size(T)[2]
num_states = size(T)[1]
terminal_states = Set([501]) # Final state is the terminal state
gamma = 0.95
k = 1


### POMDP Formulation
Transition = zeros(num_states, num_actions*2, num_states) # |S| x |A| x |S'|, T[sp, a, s] = p(sp | a, s)
Transition[:,1:num_actions,:] = deepcopy(T)
Transition[:,num_actions+1:num_actions*2,:] = deepcopy(T)

O = zeros(num_states+1, num_actions*2, num_states) # |O| x |A| x |S'|, O[o, a, sp] = p(o | a, sp)
for a in 1:num_actions
     O[1:num_states, a, 1:num_states] = Matrix{Float64}(I, num_states, num_states) ## First num_actions are SENSE actions
 end
O[num_states+1,num_actions+1:num_actions*2, 1:num_states] .= 1.0  # Next num_actions are Blind actions
     
discount = 0.95
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

    # initialstate = Deterministic(241),
    initialstate = Uniform(valid_indices),
    # initialstate = Uniform(1:num_states-1),
    isterminal = s -> s in terminal_states

)


##SARSOP
custom_lower = NativeSARSOP.BlindLowerBound(bel_res=1e-4)  # Modify parameters as needed
custom_upper = NativeSARSOP.FastInformedBound(bel_res=1e-4)  # Modify parameters as needed
sarsop_solver = SARSOPSolver(; max_time= 100.0, precision=1e-3, delta=1e-1, epsilon = 0.5, kappa = 0.5, prunethresh = 0.1, init_lower = custom_lower, init_upper = custom_upper)
start = time()
policy = solve(sarsop_solver, tabular_pomdp)
end_time = time()
println(end_time - start)

val_pol = Dict{Int, String}()
start = time()
for i in 1:num_states
    b = zeros(num_states)
    b[i] = 1.0
    b = b'
    a = action(policy, b)
    str = ""
    iterr = 0
    while(a>6 && iterr<200)
        b = b*Transition[:,a,:]
        str *= string(a-7)
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
    # Slice the first 20 characters of the string
    if !(i in terminal_states)
        push!(list, val_pol[i][1:min(end, 200)])  # Add sliced value to the list
    else
        push!(list, "")  # Add an empty string for terminal states
    end
end

script_dir = @__DIR__
file_dir = joinpath(script_dir, "SARSOP")
mkpath(file_dir)
file_path = joinpath(file_dir, "SARSOP_taxi_1.json")
open(file_path, "w") do file
    JSON.print(file, list)
end