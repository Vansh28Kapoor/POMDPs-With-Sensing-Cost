# State sensing for blind MDP agents
A Partially Observable MDP is a generalization of the MDP setting where the system dynamics are determined
by an MDP, but the agent cannot directly observe the underlying state. In this setting, the agent is aware of
its initial state but is unaware of their exact state after taking any action, except when it pays a fixed cost
K, and the state is revealed to the agent. An action where the agent does not query its state is called a blind
action, and the agent is said to be in a blind state. We apply an additional constraint that the agent can
take at most r blind actions, i.e., after r consecutive blind actions, the agent is forced to pay price K and query
its state. We must find an efficient algorithm to evaluate the optimal policy for the agent in this setting.


[Report](Report.pdf)

# generate_mdp.py

The `generate_mdp.py` script is used to generate a Markov Decision Process (MDP) based on a given scenario. The MDP is constructed considering the history of actions taken up to a specified window length, without sensing the state.

## Usage

To run the script, use the following command:

```
python generate_mdp.py [arguments]
```

## Arguments

- `--K`: (Optional) A floating-point value representing the cost to sense the state. Default is `0.1`.
    ```
    python generate_mdp.py --K 0.5
    ```

- `--window_len`: (Optional) An integer representing the maximum number of steps the agent can take without sensing. Default is `1`.
    ```
    python generate_mdp.py --window_len 3
    ```

- `--seed`: (Optional) An integer used to seed the random number generator for reproducibility. Default is `0`. If set to `-1`, the MDP parameters will be initialized from the specified `mdp_params` file.
    ```
    python generate_mdp.py --seed 42
    ```

- `--mdp_params`: (Optional) A string representing the path to a text file which contains the MDP parameters (`Cr`, `Cb`, `Tr`, `Tb`). This is used when the `seed` is set to `-1`.
    ```
    python generate_mdp.py --seed -1 --mdp_params /path/to/mdp_params.txt
    ```

## MDP Parameters File Format

If you're using the `--mdp_params` argument, ensure the file has the following format:

```
Cr: [value1, value2]
Cb: [value1, value2]
Tr: [[value1, value2], [value3, value4]]
Tb: [[value1, value2], [value3, value4]]
```

For example:

```
Cr: [0.5, 1.5]
Cb: [1.5, 0.5]
Tr: [[0.7, 0.3], [0.2, 0.8]]
Tb: [[0.3, 0.7], [0.9, 0.1]]
```

---

# MDP Planner README

---

## Introduction:

`planner.py` is a script designed to compute and evaluate policies for a specified Markov Decision Process (MDP).

## Usage:

To use the planner, run:

```
python planner.py --mdp <path_to_mdp_file> [options]
```

## Command-line Arguments:

- `--mdp`: Specifies the path to the MDP file. This argument is **required**.

- `--policy`: Path to an existing policy file that will be evaluated. (Optional)

- `--optimal`: Use this flag if you want to compute the optimal policy. (Optional)

- `--window_len`: Specifies the window length for the MDP. Default is `-1`. (Optional)

- `--print_all`: Use this flag to print the policy for all states, not just the primary ones. (Optional)

## Example:

To compute and print the optimal policy for a specific MDP:

```
python planner.py --mdp /path/to/mdp.txt --optimal
```

To evaluate an existing policy:

```
python planner.py --mdp /path/to/mdp.txt --policy /path/to/policy.txt
```

---