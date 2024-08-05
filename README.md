# MDP Planning with State Sensing Costs

In many practical sequential decision-making problems, tracking the state of the environment incurs a sensing/communication/computation cost. In these settings, the agent's interaction with its environment includes the additional component of deciding *when* to sense the state, in a manner that balances the value associated with optimal (state-specific) actions and the cost of sensing. We pose this problem as a classical discounted cost MDP with an expanded (countably infinite) state space. While computing the optimal policy for this MDP is intractable in general, we bound the sub-optimality gap associated with optimal policies in a restricted class, where the number of consecutive non-sensing (a.k.a., blind) actions is capped. We also design a computationally efficient heuristic algorithm based on policy improvement, which in practice performs close to the optimal policy.


This repository implements all the results mentioned in the paper and numerically evaluates all our results via a case study based on inventory management.
***

1. Heuristic Policy: ``Heuristic.py`` 
2. Theorem 2 (Sensing Cost Threshold): ``0Thm.py``
3. Theorem 3 (One-Step Optimality): ``1Thm.py``
4. Theorem 4 & 5 (Optimality Condition & Sub-optimality Gap): ``Thm_verif.py``
5. Case Study on Inventory Management: ``Case_Study.py`` & ``Inventory.py``


### generate_mdp.py

The `generate_mdp.py` script is used to generate a Markov Decision Process (MDP) based on a given scenario. The MDP is constructed considering the history of actions taken up to a specified window length, without sensing the state.

#### Usage

To run the script, use the following command:

```
python generate_mdp.py [arguments]
```

#### Arguments

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

#### MDP Parameters File Format

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

### MDP Planner README

---

#### Introduction:

`planner.py` is a script designed to compute and evaluate policies for a specified Markov Decision Process (MDP).

#### Usage:

To use the planner, run:

```
python planner.py --mdp <path_to_mdp_file> [options]
```

#### Command-line Arguments:

- `--mdp`: Specifies the path to the MDP file. This argument is **required**.

- `--policy`: Path to an existing policy file that will be evaluated. (Optional)

- `--optimal`: Use this flag if you want to compute the optimal policy. (Optional)

- `--window_len`: Specifies the window length for the MDP. Default is `-1`. (Optional)

- `--print_all`: Use this flag to print the policy for all states, not just the primary ones. (Optional)

#### Example:

To compute and print the optimal policy for a specific MDP:

```
python planner.py --mdp /path/to/mdp.txt --optimal
```

To evaluate an existing policy:

```
python planner.py --mdp /path/to/mdp.txt --policy /path/to/policy.txt
```

---
