# Spring 2021, IOC 5269 Reinforcement Learning
# HW2: REINFORCE with baseline and A2C

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

writer = SummaryWriter("./tb_record_1")
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_layer_1 = nn.Linear(self.observation_dim, self.hidden_size)
        nn.init.kaiming_normal_(self.shared_layer_1.weight)
        self.shared_layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.kaiming_normal_(self.shared_layer_2.weight)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        nn.init.kaiming_normal_(self.action_layer.weight)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        nn.init.kaiming_normal_(self.value_layer.weight)

        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        x = F.relu(self.shared_layer_1(state))
        x = F.relu(self.shared_layer_2(x))
        action_prob = F.softmax(self.action_layer(x), dim = -1)
        state_value = self.value_layer(x)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        state = torch.from_numpy(state).float()
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()

        ########## END OF YOUR CODE ##########

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """

        # Initialize the lists and variables
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        gamma_t = 1
        for (log_prob, state_value), reward, next_state, done in zip(saved_actions, self.rewards, self.next_states, self.dones):
            TD_target = 0.0
            if done:
                TD_target = reward
            else:
                next_state = torch.from_numpy(next_state).float()
                _, next_state_value = self.forward(next_state)
                TD_target = reward + gamma * next_state_value.item()
            policy_loss = -gamma_t * (TD_target - state_value.detach()) * log_prob
            policy_losses.append(policy_loss)
            gamma_t *= gamma
            value_loss = F.mse_loss(state_value, torch.tensor([TD_target]).float())
            value_losses.append(value_loss)

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        ########## END OF YOUR CODE ##########

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.next_states[:]
        del self.dones[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode,
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''

    # Instantiate the policy model and the optimizer
    model = Policy()
    writer = SummaryWriter("./log/")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler
        scheduler.step()

        # For each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        ########## YOUR CODE HERE (10-15 lines) ##########
        for t in range(10000):
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            model.rewards.append(reward/200)
            model.next_states.append(next_state)
            model.dones.append(done)
            ep_reward += reward
            if done:
                break
            state = next_state

        optimizer.zero_grad()
        loss = model.calculate_loss()
        writer.add_scalar("Loss/Loss", loss.item(), i_episode)
        writer.add_scalar("Reward/EP_reward", ep_reward, i_episode)
        writer.add_scalar("Reward/EWMA_reward", ewma_reward, i_episode)
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ########## END OF YOUR CODE ##########

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        writer.add_scalar("Reward/EP_reward", ep_reward, i_episode)
        writer.add_scalar("Reward/EWMA_reward", ewma_reward, i_episode)

        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold:
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''
    model = Policy()

    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    render = True

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20
    lr = 0.0005
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test('LunarLander_0.0005.pth')
