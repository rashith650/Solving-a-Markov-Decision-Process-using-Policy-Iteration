# 3 Solving a Markov Decision Process using Policy Iteration
# Name : MOHAMED RASHITH S
# Reg no : 212223243003
## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```
import gymnasium as gym
import numpy as np

# Create FrozenLake Environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8


# ----------------------------
# Policy Evaluation
# ----------------------------
def policy_evaluation(policy, env, gamma=0.99, theta=1e-8):
    V = np.zeros(n_states)

    while True:
        delta = 0

        for s in range(n_states):
            v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            delta = max(delta, abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return V


# ----------------------------
# Policy Improvement
# ----------------------------
def policy_improvement(V, env, gamma=0.99):
    policy = np.zeros((n_states, n_actions))

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        best_action = np.argmax(action_values)
        policy[s][best_action] = 1.0

    return policy


# ----------------------------
# Policy Iteration
# ----------------------------
def policy_iteration(env, gamma=0.99, theta=1e-8):

    policy = np.ones((n_states, n_actions)) / n_actions

    while True:
        V = policy_evaluation(policy, env, gamma, theta)

        new_policy = policy_improvement(V, env, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V


# ----------------------------
# Display Functions
# ----------------------------
def print_value_function(V):
    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4, 4), 4))


def print_policy(policy):

    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)

    policy_grid = np.array(
        [action_symbols[a] for a in best_actions]
    ).reshape(4, 4)

    print("\nOptimal Policy:")
    print(policy_grid)


# ----------------------------
# Run Policy Iteration
# ----------------------------
optimal_policy, optimal_value_function = policy_iteration(
    env,
    gamma=gamma,
    theta=theta
)

print_value_function(optimal_value_function)
print_policy(optimal_policy)
```

## Output

<img width="300" height="257" alt="image" src="https://github.com/user-attachments/assets/860687bd-8ec3-43e8-9d74-81b58dfd02fb" />



## Result
The Policy Iteration algorithm was successfully implemented on the FrozenLake-v1 environment using Gymnasium. The agent repeatedly performed policy evaluation and policy improvement until the policy converged. The final output includes the optimal state-value function and the optimal policy for navigating the 4×4 FrozenLake environment. The learned policy maximizes the probability of reaching the goal while avoiding holes, even in the stochastic (slippery) environment.

## Inference
Policy Iteration effectively computes the optimal policy by alternating between evaluating the current policy and improving it based on the estimated value function. Compared to Value Iteration, it generally converges in fewer iterations, although each iteration requires more computation due to the policy evaluation step. This implementation demonstrates the application of Dynamic Programming in Reinforcement Learning and highlights how an agent can learn an optimal sequence of actions in a discrete Markov Decision Process (MDP) without interacting with the environment through trial and error.
