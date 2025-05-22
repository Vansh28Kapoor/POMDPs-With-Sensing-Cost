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

file_path = joinpath(@__DIR__, "..", "Sepsis_params.npz")
data = npzread(file_path)
C = data["C"]
R = C .* -1
T = data["T"]
T = permutedims(T, (2, 1, 3))
num_actions = size(T)[2]
num_states = size(T)[1]
initial_state_array = data["init"]
initial_state_dist = Categorical(initial_state_array)
terminal_states = Set([716]) # Final state is the terminal state
gamma = 0.99

# POMDP Formulation
Transition = zeros(num_states, num_actions*2, num_states) # |S| x |A| x |S'|, T[sp, a, s] = p(sp | a, s)
Transition[:,1:num_actions,:] = deepcopy(T)
Transition[:,num_actions+1:num_actions*2,:] = deepcopy(T)
function generate_action_sequences(k::Float64)
    # Load data from .npz file
    k = k

    O = zeros(num_states+1, num_actions*2, num_states) # |O| x |A| x |S'|, O[o, a, sp] = p(o | a, sp)
    for a in 1:num_actions
        O[1:num_states, a, 1:num_states] = Matrix{Float64}(I, num_states, num_states) ## First num_actions are SENSE actions
    end
    O[num_states+1,num_actions+1:num_actions*2, 1:num_states] .= 1.0  # Next num_actions are Blind actions

    discount = 0.99
    Reward = zeros(num_states, num_actions*2)
    Reward[:,1:num_actions] = deepcopy(R)
    Reward[:,num_actions+1:num_actions*2] = deepcopy(R)
    Reward[:,1:num_actions] .-= k

    # Define the QuickPOMDP
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

        initialstate = initial_state_dist,
        isterminal = s -> s in terminal_states
    )
    return tabular_pomdp
end

function Policy(policy)
    val_pol = Vector{Vector{String}}(undef, num_states)
    total_time = 0.0
    for i in 1:num_states
        b = zeros(num_states)
        b[i] = 1.0
        b = b'  # Transpose to row vector (1xnum_states)
        # Initialize action sequence as an empty vector of strings
        action_sequence = String[]
        
        a = action(policy, b)
        iterr = 0
        
        while a > num_actions && iterr < 500
            start = time()
            b = b * Transition[:, a, :]
            end_time = time()
            total_time += end_time - start
            push!(action_sequence, string(a - num_actions -1))
            a = action(policy, b)
            iterr += 1
        end
        
        # Append final adjusted action (a-1) as a string
        push!(action_sequence, string(a - 1))
        if i%100 ==0
            println("State Completed_$(i)", "Itteration Time:", end_time - start, "Total Time:", total_time)
            flush(stdout)
        end
        # Store the action sequence
        val_pol[i] = deepcopy(action_sequence)
    end
    return val_pol
end

if abspath(PROGRAM_FILE) == @__FILE__
    custom_lower = NativeSARSOP.BlindLowerBound(bel_res=1e-4)  # Modify parameters as needed
    custom_upper = NativeSARSOP.FastInformedBound(bel_res=1e-4)  # Modify parameters as needed
    sarsop_solver = SARSOPSolver(; max_time= 3000.0, precision=1e-3, delta=1e-1, epsilon = 0.5, kappa = 0.5, prunethresh = 0.1, init_lower = custom_lower, init_upper = custom_upper)
    for k in [0.1, 0.05, 0.01, 0.005]
        pomdp = generate_action_sequences(k)
        start = time()
        policy = solve(sarsop_solver, pomdp)
        lst = Policy(policy)

        script_dir = @__DIR__
        file_dir = joinpath(script_dir, "SARSOP")
        mkpath(file_dir)
        file_path = joinpath(file_dir, "Sepsis_500_$(k).json")
        open(file_path, "w") do file
            JSON.print(file, lst)
        end
    end

end