# MDP Planning with State Sensing Costs

In many practical sequential decision-making problems, tracking the state of the environment incurs a sensing/communication/computation cost. In these settings, the agent's interaction with its environment includes the additional component of deciding *when* to sense the state, in a manner that balances the value associated with optimal (state-specific) actions and the cost of sensing. We pose this problem as a classical discounted cost MDP with an expanded (countably infinite) state space. While computing the optimal policy for this MDP is intractable in general, we bound the sub-optimality gap associated with optimal policies in a restricted class, where the number of consecutive non-sensing (a.k.a., blind) actions is capped. We also design a computationally efficient heuristic algorithm based on policy improvement, which in practice performs close to the optimal policy.


This repository implements all the results mentioned in the paper and numerically evaluates all our results via a case study based on inventory management.
***

1. Heuristic Policy: ``Heuristic.py`` 
2. Theorem 2 (Sensing Cost Threshold): ``Sensing_Threshold.py``
3. Lemma 3 (One-Step Optimality): ``OneStep_Opt.py``
4. Theorem 4 (Optimality Condition & Sub-optimality Gap): ``Thm_verif.py``
5. Case Study on Inventory Management: ``Case_Study.py`` & ``Inventory.py``

## `Heuristic.py`

`Heuristic.py` implements the heuristic algorithm with the following variables:

1. **`T`** (Transition Probability Matrix): This is a $|A| \times |S| \times |S|$ array, where each slice along the first dimension represents the transition matrix for a specific action. For instance, `T[a, s1, s2]` denotes the probability of transitioning from state `s1` to state `s2` by playing action `a`.
2. **`C`** (Cost Matrix): This is a $|S| \times |A|$ array, where each column represents the cost incurred for each action across the states.
3. **`V`** (Optimal Value Function without Sensing Cost): This is a $|S|$ array denoting the optimal value function for each state.
4. **`gamma`** (Discounting Factor $\alpha$): The discounting factor $\alpha$ of the MDP.
5. **`k`** (Sensing Cost $k$): The state sensing cost $k$ for the MDP.

The function `Heuristic` returns a tuple containing:

1. **Value Function**: The value function corresponding to the heuristic policy for each state.
2. **Policy**: A list where each entry is a string representing the sequence of actions to be taken for each root state. The `max` parameter of `Heuristic` limits the maximum length of these strings to handle cases where no sensing is applied starting from the root state.

To implement the heuristic algorithm, execute `Heuristic.py` with the variables set to the desired MDP parameters as mentioned above (edit the file accordingly).

## `Sensing_Threshold.py`

`Sensing_Threshold.py` evaluates the sensing cost threshold algorithm using the following variables:

1. **`actions`**: A list of actions in the Baseline MDP.
2. **`numHeadStates`** ($|S|$): The number of root states in the MDP.
3. **`gamma`** (Discounting Factor $\alpha$): The discount factor used in the MDP.
4. **`T`** (Transition Probability Matrix): A dictionary where the keys are actions, and the values are arrays representing the transition matrix for each action.
5. **`C`** (Cost Matrix): A dictionary where the keys are actions, and the values are arrays representing the cost incurred for each action across the states.

For instance, the values for a 2-state, 2-action MDP are:

```python
actions = ["R", "B"]

T = {
    "R": np.array([[0.7, 0.3], [0.2, 0.8]]),
    "B": np.array([[0.3, 0.7], [0.9, 0.1]])
}

C = {
    "R": np.array([0.25, 0.75]),
    "B": np.array([1, 0.5])
}
```
To evaluate the sensing cost threshold, execute `Sensing_Threshold.py` with the variables set to the desired MDP parameters as mentioned above (edit the file accordingly). The script will print the sensing cost threshold in the format: `Sensing Cost Threshold: <value>`.

## `OneStep_Opt.py`

`OneStep_Opt.py` verifies the One-Step Optimality condition for each of the root states and returns an array of boolean values for each state. It utilizes all the variables: `actions`, `numHeadStates`, `gamma`, `T`, and `C` exactly as described for **`Sensing_Threshold.py`**, along with:

1. **`windowLength`** (Truncated MDP depth $N$): The depth $N$ of the truncated MDP for which the theorem is being applied.
2. **`sensingcost`** (Sensing Cost $k$): The state sensing cost $k$ for the MDP.

To determine whether Lemma 3 is satisfied, execute `OneStep_Opt.py`. The script will print the output in the following format: <br>
``Sensing Cost: <k>, Window_len: <N>``<br>
``LHS_min:`` $[min_{i \in \mathcal{L}^j_{N+1}}G_{N+1}(j,i) \ \text{for} \ j \in \mathcal{S}]$, ``Constrained Value Fn:`` $[V_{\mathcal{M}_{k,N}}(j) \ \text{for} \ j \in \mathcal{S}]$ <br>
[`<bool(j)>` $\ \text{for} \ j \in \mathcal{S}$]



## `Thm_verif.py`

Thm_verif.py evaluates the bound on the sub-optimality gap between the optimal value function for $𝓜_{k}$ and that of the truncated MDP $𝓜_{k,N}$, and also verifies the optimality of the optimal policy for the truncated MDP. It utilizes exactly the same variables as `OneStep_Opt.py`: `actions`, `numHeadStates`, `gamma`, `T`, `C`, `windowLength`, and `sensingcost`. The script will print the output in the following format: <br>
``Sensing Cost: <k>, Window_len: <N>``<br>
``Suboptimality Gap: <arr>`` <br>
``<arr>`` is an array where each element corresponds to the bound on the sub-optimality gap. If ``<arr>`` is an array of zeros, it implies that Theorem 4 (Optimality Theorem) criterion is satisfied. To execute the script, set the variables to the desired MDP parameters as mentioned above and run `Sensing_Threshold.py` (edit the mentioned variables in the file as needed).

## `Inventory.py`

`Inventory.py` evaluates the transition probability matrix and cost matrix based on the following inventory parameters:

1. **`inv_capacity`**: The maximum number of items that can be stored in the inventory.
2. **`actions`**: The total number of actions (number of items to be produced each month) or the cardinality of the action space.
3. **`prob_demand`**: A list representing the probability distribution of the demand.
4. **`holding_cost`**: The inventory-carrying cost per unit leftover at the end of the month.
5. **`production_cost`**: The production cost per unit of the item.
6. **`profit`**: The selling price per unit of the item.


The script prints the output in the following format: <br>
``Cost Matrix: <arr1>, Probability Matrix: <arr2>`` <br>

`<arr1>` is $|A| \times |S|$ array where each row represents the cost incurred for each action across the states and `<arr2>` is the usual $|A| \times |S| \times |S|$ array, where each slice along the first dimension represents the transition matrix for a specific action.


## `Case_Study.py`
`Case_Study.py` is used to evaluate the optimal value function and policy for the truncated MDP parameters (Eg: `Inventory.py` parameters). It utilizes exactly the same variables as `OneStep_Opt.py`: `actions`, `numHeadStates`, `gamma`, `T`, `C`, `windowLength`, and `sensingcost`. The script will print the output in the following format: <br>
`Window_len: <N>` <br>
⋮ <br>
`state <i>, policy: <tuple(strings)>, value: <value_fun>` <br>
⋮ <br>


***
