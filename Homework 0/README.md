# Fundamentals – MDPs, Policy Iteration, and Value Iteration

## Problem 1 (Q-Value Iteration)

[HackMD](https://hackmd.io/5fAyhbG1SOWrcmji2e_0kQ#Problem-1-Q-Value-Iteration)

## Problem 2 (Soft Policy Iteration for Regularized MDPs)

[HackMD](https://hackmd.io/5fAyhbG1SOWrcmji2e_0kQ#Problem-2-Soft-Policy-Iteration-for-Regularized-MDPs)

## Problem 3 (Implementing Policy Iteration and Value Iteration) 

In this problem, we will implement policy iteration and value iteration for a classic MDP environment called “Taxi” (Dietterich, 2000). 

This environment has been included in the OpenAI Gym: https://www.gymlibrary.dev/environments/toy_text/taxi/.

The action map for taxi is as follows:

| Action | Meaning |
| :---: | :---: |
| 0 | move south |
| 1 | move north |
| 2 | move east |
| 3 | move west |
| 4 | pickup passenger |
| 5 | dropoff passenger |

The reward map for taxi is as follows:

| Reward | Meaning |
| :---: | :---: |
| -10 | illegal action |
| -1 | per time step |
| 20 | dropoff passenger |