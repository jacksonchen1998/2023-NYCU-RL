# Policy Gradient and Model-Free Prediction

## Problem 1 (Baseline for Variance Reduction)

[Report](./Spring2023_RL_HW1_311511052.pdf)

## Problem 2 (Policy Gradient)

[Report](./Spring2023_RL_HW1_311511052.pdf)

## Problem 3 (Monte Carlo Policy Evaluation)

[Report](./Spring2023_RL_HW1_311511052.pdf)

## Problem 4 (Policy Gradient Algorithms With Function Approximation)

<table>
    <tr>
        <th>
            Task
        </th>
        <th>
            GIF
        </th>
    </tr>
    <tr>
        <td>
            CartPole-v0
        </td>
        <td>
            <img src="./gif/task1.gif" width="500" height="300">
        </td>
    </tr>
    <tr>
        <td>
            LunarLander-v2
        </td>
        <td>
            <img src="./gif/task2.gif" width="500" height="300">
        </td>
    </tr>
</table>

### REINFORCE

<img src = "./image/reinforce.png">

### REINFORCE with Baseline

<img src = "./image/reinforce_baseline.png">

### REINFORCE with Generalized Advantage Estimation

$\lambda = 0.99$

<img src = "./image/reinforce_gae_099_.png">

$\lambda = 0.98$

<img src = "./image/reinforce_gae_098_.png">

$\lambda = 0.97$

<img src = "./image/reinforce_gae_097_.png">