# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import os
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

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_2")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared_first = nn.Linear(self.observation_dim, self.hidden_size) # initialize the shared layer
        nn.init.kaiming_normal_(self.shared_first.weight) # random weight initialization
        self.shared_second = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.kaiming_normal_(self.shared_second.weight)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim) # initialize the action layer
        nn.init.kaiming_normal_(self.action_head.weight)
        self.value_head = nn.Linear(self.hidden_size, 1) # initialize the value layer, since we only need one value, we set the output dimension to 1
        nn.init.kaiming_normal_(self.value_head.weight)
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        ########## YOUR CODE HERE (3~5 lines) ##########
        input = F.relu(self.shared_first(state))
        input = F.relu(self.shared_second(input))
        action_prob = F.softmax(self.action_head(input), dim=-1) # softmax over the last dimension
        state_value = self.value_head(input) # no activation function for the value layer
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
        state = torch.from_numpy(state).float() # convert the state to a tensor
        action_probability, state_value = self.forward(state) # get the action probability and the state value
        m = Categorical(action_probability) # create a categorical distribution over the list of probabilities of actions
        action = m.sample() # and sample an action using the distribution
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
        R = 0
        saved_actions = self.saved_actions # get the saved actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)

        gamma_t = 1
        for (log_prob, value), sample in zip(saved_actions, returns):
            policy_losses.append(-gamma_t * (sample - value.detach()) * log_prob) 
            # detach the value from the graph, where the value is the baseline
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([sample])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
    
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

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process

        ########## YOUR CODE HERE (10-15 lines) ##########
        for t in range(10000):
            action = model.select_action(state)
            state_prime, reward, done, _ = env.step(action)
            model.rewards.append(reward/200) # normalize the reward
            ep_reward += reward
            if done:
                break
            state = state_prime

        optimizer.zero_grad()
        loss = model.calculate_loss(0.99) 
        writer.add_scalar('Loss', loss.item(), i_episode)
        loss.backward()
        optimizer.step()
        model.clear_memory()
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))
        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Length', t, i_episode)
        writer.add_scalar('Learning Rate', scheduler.get_lr()[0], i_episode)
        writer.add_scalar('EWMA Reward', ewma_reward, i_episode)
        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander-v2_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
                # save every frame into png files
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.005
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test('LunarLander-v2_{}.pth'.format(lr))
