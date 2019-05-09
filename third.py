from keras import optimizers

import random
import gym
import gym_2048
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

env = gym.make('2048-v0')
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
state_size
action_size = env.action_space.n
action_size
batch_size = 128
n_episodes = 2000
output_dir = 'model_output/2048'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
	
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        #model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * self.model.predict(next_state))
            target_f = self.model.predict(state)
            #print(target_f[0], target)
            target_f[0] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




done = False

agent = DQNAgent(state_size, action_size)



for e in range(n_episodes):

    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(10000):
        #env.render()

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        reward = reward if not done else -10

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episode: {}/{}, max_tile: {}, epsilon: {:2}".format(e, n_episodes, max(max(state)), agent.epsilon))
            break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(output_dir + "weights_" + "{:04d}".format(e) + ".hdf5")


state = env.reset()
state = np.reshape(state, [1, state_size])
agent.gamma = 0

highest_time = 0
high_env = 0

for t in range(1000):
    env.reset()
    for time in range(10000):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

        if done:
            if time > highest_time:
                highest_time = time
                env.render()
                print("---")
            break

print("highest time: {}".format(time))
#print("best result: {}".format(env.render()))
        
