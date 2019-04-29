import gym_2048
import gym
import random 
from collections import deque
import torch
import torch.nn as nn

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
    
    def build():
        model = nn.Sequential(nn.Linear(input_size, 12),
                              nn.ReLU(),
                              nn.Linear(12, 12),
                              nn.ReLU(),
                              nn.Linear(12, output_size),
                              nn.ReLU())
        return model
    
    def action(self, state):
        if np.random.rand() < self.exploration:
            return random.randrange(self.action_size);
        return np.argmax(self.model.forward(state))
        #isnt perfect yet, not sure what self.model.forward will return, some sort of tensor?




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
            action = agent.action(state)

        #under this is taken from open ai
        action = env.np_random.choice(range(4), 1).item()
        next_state, reward, done, info = env.step(action)
        moves += 1

        print('Next Action: "{}"\n\nReward: {}'.format(
              gym_2048.Base2048Env.ACTION_STRING[action], reward))
        env.render()

  print('\nTotal Moves: {}'.format(moves))
