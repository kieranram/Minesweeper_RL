from tqdm import tqdm

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise

from minefield import Minefield

from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


class TrainingLog:
    
    def __init__(self, buffer_size = 500):
        self.states = None # full record of states in every game, excluding final state
        self.rewards = [] # reward received at each step
        self.actions = [] # actions chosen at each step
        self.is_end = [] # should we calculate value of next state? 

        self.finals = None # last board seen, end states
        self.lengths = [] # length of each game (to find end conditions)

        self.buffer = buffer_size

    
    def log_step(self, state, action, reward, done):
        if self.states is None:
            self.states = state
        else:
            self.states = np.concatenate([self.states, state], axis = 0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_end.append(done)

    def log_final(self, state, n_steps):
        if self.finals is None:
            self.finals = state
        else:
            self.finals = np.concatenate([self.finals, state], axis = 0)
        self.lengths.append(n_steps)

        self.states = self.states[-self.buffer:]
        self.actions = self.actions[-self.buffer:]
        self.rewards = self.rewards[-self.buffer:]
        self.is_end = self.is_end[-self.buffer:]

class DQ_Sweeper:
  
    def __init__(self, env, filters, sizes, eps0, eps_decay, eps_decay_freq, gamma, f = False, training_buffer = 500):
        
        self.env = env
        
        self.env_shape = (env.n_rows, env.n_cols, 11)
        if f:
            self.model = keras.models.load_model(f)
        else:
            self.make_model(filters, sizes)
        
        self.epsilon = eps0
        self.eps_decay = eps_decay
        self.eps_freq = eps_decay_freq
        
        self.gamma = gamma
        
        self.logger = TrainingLog()
        
        
    def make_model(self, filters, sizes):
        inputs = keras.Input(shape = self.env_shape)
        print(inputs.shape)

        conv1 = Conv2D(filters[0], sizes[0], padding = 'same', activation = 'relu', input_shape = self.env_shape, data_format = 'channels_last')(inputs)
        conv1 = Conv2D(filters[0], sizes[0], padding = 'same', activation = 'relu')(conv1)
        conv1 = GaussianNoise(0.2)(conv1)
        print(conv1.shape)
        conv2 = Conv2D(filters[1], sizes[1],  padding = 'same', activation = 'relu')(conv1)
        conv2 = GaussianNoise(0.2)(conv2)
        print(conv2.shape)

        conv3 = Conv2D(filters[2], sizes[2], padding = 'same', activation = 'relu')(conv2)
        conv3 = GaussianNoise(0.2)(conv3)

        conv4 = Conv2D(3, 1, padding = 'same', activation = 'relu')(conv3)
        output = Conv2D(1, 3, padding = 'same', activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1))(conv4)
        print(output.shape)

        model = keras.Model(inputs = inputs, outputs = output)
        model.compile('adam', loss='mse')
        print('Model Made')
        
        self.model = model
        
    def choose_action(self, state, use_eps = True):
      
        if (np.random.random() < self.epsilon) and use_eps: 
            action_row = np.random.randint(self.env.n_rows)
            action_col = np.random.randint(self.env.n_cols)
        else: 
            Qs = self.model.predict(state)
            Qs = Qs.reshape(self.env_shape[0], self.env_shape[1])
            action_row, action_col= np.where(Qs == Qs.max())
            action_row, action_col = action_row[0], action_col[0]

        action = (action_row, action_col)

        return action
        
    def run_episode(self, log = True, use_epsilon = True):
        self.env.prime()
        
        current_state = self.env.network_obs
        steps = 0
        
        while True:
            action = self.choose_action(current_state, use_eps = use_epsilon)
            new_state, reward, done, _ = self.env.step(action)
            steps += 1

            if log:
                self.logger.log_step(current_state, action, reward, done)

            if done: 
                if log:
                    self.logger.log_final(new_state, steps)
                break

            current_state = new_state

    def fit(self, batch_size):
        n_hist = self.logger.states.shape[0]
        if n_hist < 3 * batch_size:
            return

        samps = np.random.choice(n_hist, size = batch_size, replace = False)

        states = self.logger.states[samps]
        actions = [self.logger.actions[i] for i in samps]

        rewards = np.array([self.logger.rewards[i] for i in samps])

        is_end = np.array([self.logger.is_end[i] for i in samps])

        preds = self.model.predict(states)

        # mod to deal with case of last sample being chosen
        q_maxes = self.model.predict(self.logger.states[(samps + 1) % n_hist]).max(axis = (1, 2, 3)) 
        adj = q_maxes * (1 - is_end) # if not the final move, add value of next state
        vals = rewards + adj

        for i in range(batch_size):
            x, y = actions[i]

            preds[i, x, y, 0] = vals[i]

        self.model.fit(states, preds, verbose = False)

    def show_single(self, seed = None):
        np.random.seed(seed)
        self.env.prime()

        current_state = self.env.network_obs
        steps = 0

        states = []
        vals = []

        while True:
            pred = self.model.predict(current_state).reshape(self.env_shape[0], self.env_shape[1])
            row, col = np.where(pred == pred.max())

            states.append(self.env.last_obs)
            vals.append(pred)
            row, col = row[0], col[0]
            new_state, _, done, _ = self.env.step((row, col))
            steps += 1

            if done:
                last_val = self.model.predict(new_state).reshape(self.env_shape[0], self.env_shape[1])
                vals.append(last_val)
                break

            if steps > 40:
                break
            current_state = new_state

        plt.figure(figsize = (10, 5 * steps))

        for i in range(steps):
            l = plt.subplot(steps, 2, 2 * i + 1)
            r = plt.subplot(steps, 2, 2 * i + 2)
            
            sns.heatmap(states[i], ax = l, annot = True, cbar = False, vmin = -1, vmax = 5)
            sns.heatmap(vals[i], ax = r, cbar = False)

        plt.show()





