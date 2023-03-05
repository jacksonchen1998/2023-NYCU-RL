# Fundamentals – MDPs, Policy Iteration, and Value Iteration

## Problem 1 (Q-Value Iteration)

**(a)** Define $V_*(s) := \max_\pi V^\pi(s)$ and $Q_*(s,a)$. Suppose $\gamma \in (0,1)$.

Prove the following Bellman optimality equations:
$$
\begin{align}
    V_*(s) = \max_a Q_*(s,a) \nonumber \\
    Q_*(s,a) = R_s^a + \gamma \cdot\sum_{s^\prime} P^a_{ss^\prime}V_*(s^\prime) \nonumber
\end{align}
$$

---

A:

Firstly, we prove $V_*(s) = \max_a Q_*(s,a)$
$$
    \because V^\pi(s) = \sum_a \pi(a \mid s)Q^\pi(s,a) \\
\begin{align}
\therefore V_*(s) &
= \max_\pi(\sum_a \pi(a \mid s)Q^\pi(s,a)) \le \max_\pi(\sum_a\pi(a \mid s)Q^*(s,a)) \nonumber \\
&=Q_*(s,a) \le \max_a(Q_*(s,a))
\end{align}
$$
Suppose that $V_* \lt \max_aQ_*(s,a)$ happened when $s = s_t$ and $Q^{\pi_*}(s,a) \equiv Q_*(s,a)$
As we know the theorem one-step policy improvement, we can get 
$$
\begin{align}
    \forall s, \ \ \pi^\prime = \arg \max_a Q^{\pi_*}(s,a)
\end{align}
$$

Before proving $V_*(s) = \max_a Q_*(s,a)$, we assume that $\pi^\prime$ is deterministic.

$$
\begin{align}
V_*(s) &= \max_\pi V^{\pi}(s) \ge V^{\pi^\prime}(s) \nonumber \\
&= Q^{\pi^\prime}(s, \pi^\prime(s)) \nonumber \ \ (\mathrm{Bellman \ Equation}) \\
&= R_s^{\pi^\prime(s)} + \gamma \sum_{s^\prime} P(s^\prime \mid s , \pi^\prime(s)) \cdot V^{\pi^\prime}(s^\prime) \nonumber \ \ \ (\mathrm{montonic \ policy \ improvement}) \nonumber \\
&\ge R^{\pi^\prime(s)}_s + \gamma \sum_{s^\prime} P(s^\prime \mid \pi^\prime(s)) \cdot V^{\pi_*}(s^\prime) \nonumber \ \ \ (\mathrm{by}(2)) \\
&= \max_a Q^{\pi_*}(s,a) \nonumber
\end{align}
$$ Since $(1)$ we can get $\max_a Q_* (s,a) \gt V_*(s)$. We found a contradictor that $\max_a Q_* (s,a) > V_*(s) = \max_a Q_* (s,a)$

$\therefore$ We can get the equation $V_*(s) = \max_a Q_*(s,a)$

And then we can prove $ Q_*(s,a) = R_s^a + \gamma \sum_{s^\prime} P^a_{ss^\prime}V_*(s^\prime)$

$$
\begin{align}
    Q_*(s,a) &= \max_\pi Q^\pi (s,a) \nonumber \\
    &= \max_\pi R_s^a + \gamma \cdot \sum_{s^\prime} P^a_{ss^\prime} \cdot V^\pi(s^\prime) \nonumber \\
    &\le R_s^a + \gamma \cdot \sum_{s^\prime} P^a_{ss^\prime} \cdot V_* (s^\prime)
\end{align}
$$

Since $(3)$, we can define policy $\pi^\prime$ as : 

$$
    \forall s, \ \ \pi^\prime = \arg \max_a Q^{\pi_*}(s,a)
$$

As we know $V_*(s) = V^{\pi^\prime}$, for any $s$

$$
\begin{align}
    \therefore R^a_s + \gamma \cdot \sum_{s^\prime} P_{ss^\prime}^a \cdot V_*(s^\prime) &= R_s^a + \gamma \cdot \sum_{s^\prime} P^a_{ss^\prime} \cdot V^{\pi^\prime}(s) \nonumber \ \ \ (\mathrm{def.} \ Q^\pi \mathrm{and} \ V^\pi ) \\
    &= Q^{\pi^\prime} (s,a) \nonumber \\
    &\le \max_\pi Q^\pi (s,a) \nonumber \\
    &= Q_* (s,a) \ \ \ \mathrm{\mathrm{def.} \ Q_* (s,a)}
\end{align}
$$ By the equation $(3)$ and $(4)$, we can get the equation $Q_*(s,a) = R_s^a + \gamma \cdot\sum_{s^\prime} P^a_{ss^\prime}V_*(s^\prime)$

---

**(b)** Based on (a), we thereby have the recursive Bellman optimality equation for the optimal action-value function $Q_*$ as:
$$
Q_*(s, a)=R_s^a+\gamma \sum_{s^{\prime}} P_{s s^{\prime}}^a\left(\max _{a^{\prime}} Q_*\left(s^{\prime}, a^{\prime}\right)\right)
$$ Similar to the standard Value Iteration, we can also study the Q-Value Iteration by defining the Bellman optimality operator $T^*: \mathbb{R}^{|\mathcal{S}||\mathcal{A}|} \rightarrow \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ for the action-value function: for every state-action pair $(s, a)$
$$
\left[T^*(Q)\right](s, a):=R_s^a+\gamma \sum_{s^{\prime}} P_{s s^{\prime}}^a \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}\right)
$$ Show that the operator $T^*$ is a $\gamma$-contraction operator in terms of $\infty$-norm.

---

A:

$$
\begin{align}
    \Vert T^*Q - T^* \widehat{Q}\Vert_\infty &= \max_{(s,a)} \vert [T^*Q](s,a) - [T^* \widehat{Q}](s,a) \vert \nonumber \\
    &= \max_{(s,a)} \vert (R_s^a) + \gamma \sum_{s^\prime} P^a_{ss^\prime} \cdot \max_{a^\prime} Q(s^\prime, a^\prime) - (R_s^a + \gamma \sum_{s^\prime} P_{ss^\prime}^a \cdot \max_{a^{\prime\prime}} \widehat{Q}(s^\prime, a^{\prime\prime}) )\vert \nonumber \\
    &= \max_{(s,a)} \gamma \vert \sum_{s^\prime} P_{ss^\prime}^a \cdot (\max_{a^\prime} Q(s^\prime, a^\prime) - \max_{a^{\prime\prime}} \widehat{Q}(s^\prime, a^{\prime\prime}))\vert \nonumber \\
    &\le \max_{(s,a)} \gamma \cdot \sum_{s^\prime} P^a_{ss^\prime} \cdot \underbrace{\vert \max_{a^\prime} Q(s^\prime, a^\prime) - \max_{a^{\prime\prime}} \widehat{Q}(s^\prime, a^{\prime\prime}) \vert}_{\le \ \max_{a^\prime} \vert Q(s^\prime, a^\prime) - \widehat{Q}(s^\prime, a^{\prime\prime}) \vert} \nonumber \\
    &\le \max_{(s^\prime, a^\prime)} \gamma \cdot \sum_{s^\prime} P^a_{ss^\prime} \cdot \max_{a^\prime} \vert Q(s^\prime, a^\prime) - \widehat{Q}(s^\prime, a^{\prime\prime}) \vert \nonumber \\
    &\le \gamma \cdot \max_{(s^\prime, a^\prime)} \vert Q(s^\prime, a^\prime) - \widehat{Q}(s^\prime, a^{\prime\prime}) \vert \nonumber \\
    &= \gamma \cdot \Vert Q - \widehat{Q} \Vert_{\infty} \nonumber
\end{align}
$$

After that, $T^*$ is a $\gamma$-contraction operator with $L_\infty$-norm.

## Problem 2 (Soft Policy Iteration for Regularized MDPs)

In the $k$-th iteration, given the entropy-regularized $\mathrm{Q}$ function $Q_{\Omega}^{\pi_k}$ with $\Omega(\pi(\cdot \mid s)):=\sum_{a \in \mathcal{A}} \pi(a \mid s) \log \pi(a \mid s)$, under Soft Policy Iteration, the new policy for the $k+1$-iteration can be obtained by solving the following optimization problem for each state $s \in \mathcal{S}$ :
$$
\pi_{k+1}(\cdot \mid s)=\arg \max _\pi\left\{\left\langle\pi(\cdot \mid s), Q_{\Omega}^{\pi_k}(s, \cdot)\right\rangle-\Omega(\pi(\cdot \mid s))\right\}
$$

Please show that the optimal solution to the above optimization is
$$
\pi_{k+1}(\cdot \mid s)=\frac{\exp \left(Q_{\Omega}^{\pi_k}(s, \cdot)\right)}{\sum_{a \in \mathcal{A}} \exp \left(Q_{\Omega}^{\pi_k}(s, a)\right)}
$$

---

A:

Firstly, we can construct Lagrangian as

$$
L(\pi):=\sum_{a \in \mathcal{A}}\left(\pi(a \mid s) Q_{\Omega}^{\pi_k}(s, a)-\pi(a \mid s) \log \pi(a \mid s)\right)-\mu\left(\sum_{a \in \mathcal{A}} \pi(a \mid s)-1\right) .
$$ Then we can use $\frac{\partial L(\pi)}{\partial \pi(a \mid s)}=0$ to find the optimal solution.
$$
\begin{align}
&\Rightarrow \frac{\partial L(\pi)}{\partial \pi(a \mid s)} = (Q^{\pi_k}_\Omega(s,a) - \log \pi (a \mid s) - 1) -\mu = 0, \ \forall a \in \mathcal {A} \nonumber \\ &\Rightarrow \log \pi (a \mid s) = Q^{\pi_k}_\Omega(s,a) - \mu - 1 \nonumber \\ &\Rightarrow \pi(a \mid s) = \frac{e^{Q^{\pi_k}_\Omega(s,a)}}{e^{\mu+1}}
\end{align}
$$

And we can use $(5)$ at $\sum_{a \in \mathcal{A}} \pi (a \mid s) - 1 = 0$:
$$
\begin{align}
\sum_{a \in \mathcal{A}} \frac{e^{Q^{\pi_k}_\Omega(s,a)}}{e^{\mu+1}} = 1 \nonumber \\
\Rightarrow e^{\mu+1} = \sum_{a \in \mathcal{A}} e^{Q^{\pi_k}_\Omega(s,a)}
\end{align}
$$

And we can know that $\forall a \in \mathcal{A}$, we can change the equation $(5)$ into $(6)$

$$
\begin{align}
&\Rightarrow \frac{e^{Q^{\pi_k}_\Omega(s, \cdot)}}{\pi(\cdot \mid s)} = \sum_{a \in \mathcal{A}} e^{Q^{\pi_k}_\Omega(s,a)} \nonumber \\
&\Rightarrow \pi(\cdot \mid s) = \frac{e^{Q^{\pi_k}_\Omega(s, \cdot)}}{\sum_{a \in \mathcal{A}} e^{Q^{\pi_k}_\Omega(s,a)}} \nonumber
\end{align}
$$ 

## Problem 3 (Implementing Policy Iteration and Value Iteration) 

In this problem, we will implement policy iteration and value iteration for a classic MDP environment called “Taxi” (Dietterich, 2000). 

This environment has been included in the OpenAI Gym: https://gym.openai.com/envs/Taxi-v3/.