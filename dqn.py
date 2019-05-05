import gym_2048
import gym
import random 
from collections import deque
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#env = gym.make('2048-v0').unwrapped

'''
class NN(nn.Module):
    def __init__(self, input_size, output_size, lr):
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Sequential(nn.Linear(input_size, 12),
                              nn.ReLU(),
                              nn.Linear(12, 12),
                              nn.ReLU(),
                              nn.Linear(12, output_size),
                              nn.ReLU())
        self.loss = nn.MSELoss(reduction='sum')
        
    def predict(
'''
class DQNModel(nn.Module):
    def __init__(self, in_features, num_actions):
        '''
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        '''
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque()
        self.discount = 0.9 #discount, gamma
        self.exploration = 1.0 #exploration, epsilon
        self.exploration_decay = 0.95
        self.min_exploration = 0.01
        self.lr = 0.01
        self.model = self.build()
    
    def build(state_size, action_size):
        '''
        model = nn.Sequential(nn.Linear(input_size, 12),
                              nn.ReLU(),
                              nn.Linear(12, 12),
                              nn.ReLU(),
                              nn.Linear(12, output_size),
                              nn.ReLU())
        '''
        model = DQNModel(state_size, action_size)
        return model
    
    def action(self, state):
        if np.random.rand() < self.exploration:
            return random.randrange(self.action_size);
        return np.argmax(self.model.forward(state))
        #isnt perfect yet, not sure what self.model.forward will return, some sort of tensor?

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



if __name__ == '__main__':
    env = gym.make('2048-v0')
    env.seed(42)

    env.reset()
    env.render()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)

    done = False
    moves = 0

    for ep in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for t in range(500):
            moves += 1
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(ep, 1000, t, agent.exploration))
                break
            else:
                loss = nn.MSELoss()


                



        #under this is taken from open ai
        action = env.np_random.choice(range(4), 1).item()
        next_state, reward, done, info = env.step(action)
        moves += 1

        print('Next Action: "{}"\n\nReward: {}'.format(
              gym_2048.Base2048Env.ACTION_STRING[action], reward))
        env.render()

  print('\nTotal Moves: {}'.format(moves))
