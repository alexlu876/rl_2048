import os
import argparse
import time
import numpy as np
import json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import gym
import torch
import torch.nn as nn
import torch.optim as optim


#parser = argparse.ArgumentParser('Trainer')
#parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
#parser.add_argument('--data_size', type=int, default=1000)
#parser.add_argument('--batch_time', type=int, default=10)
#parser.add_argument('--batch_size', type=int, default=20)
#parser.add_argument('--niters', type=int, default=2000)
#parser.add_argument('--test_freq', type=int, default=20)
#parser.add_argument('--viz', action='store_true')
#parser.add_argument('--gpu', type=int, default=0)
#parser.add_argument('--adjoint', action='store_true')
#args = parser.parse_args()




class Buffer(object):
    def __init__(self, max_memory=50, discount=.9):
    	#setting the maximum depth of buffer
        self.max_memory = max_memory

        #memory stored as a python list
        self.memory = list()

        #how to weight experience further in the past
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        # this is what the memory list actually looks like
        self.memory.append([states, game_over])

        #loop to keep the memory at the desired length
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
    	#total number of experiences currently stored
        len_memory = len(self.memory)
        
        #actions corresponding
        num_actions = model.output_shape[-1]


        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, x in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[x][0]
            game_over = self.memory[x][1]
            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q
        return inputs, targets

if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration factor
    num_actions = 3  # [left, stay, right]
    epoch = 20
    max_memory = 50
    hidden_size = 100
    batch_size = 50
    input_size = 2

    x_range = [-1.2, .6]
    v_range = [-0.07, 0.07]
    start_params = [.3, 0.0]
    x_goal = 0.5

    model = ODEFunc(3, 2)

    env = env = gym.make('MountainCar-v0')

    buff = Buffer()

    #Now the actual trianing loop
    no_of_successes = 0;
    for e in range(epoch):
    	loss = 0.0
    	env.reset()
    	over = False
    	curr_input_t = env.observe()
    	curr_iter = 0
    	step = 0
    	while (game_over == False):
    		input_time = curr_input_t
            step += 1
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                action = #iLQR witn ode and state as input

            curr_input_t, reward, game_over = env.act(action)

            if (reward >= 90):
            	no_of_successes += 1;

            buffer.remember([input_time, action, reward, curr_input_t], game_over)

            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)[0]
        print("Epoch" + e + ". Step: " + step + ". Wins: " + no_of_successes + ". Loss: " + loss)
