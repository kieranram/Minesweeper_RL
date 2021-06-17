from tqdm import tqdm

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise

from minefield import Minefield

from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter 

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

        self.tests = []

    
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
  
    def __init__(self, env, filters, sizes, eps0, eps_decay, eps_decay_freq, gamma, 
                testing_freq = 100, n_test = 10, save_freq = 10000, sname = 'model',
                f = False, training_buffer = 500, min_eps = 0.01, weights = None):
        
        self.env = env
        
        self.env_shape = (env.n_rows, env.n_cols, 11)
        if f:
            self.model = keras.models.load_model(f)
        else:
            self.make_model(filters, sizes)

        if weights:
            temp_model = keras.models.load_model(weights)
            for i, layer in enumerate(temp_model.layers):
                layer_weights = layer.get_weights()
                if len(layer_weights):
                    self.model.layers[i].set_weights(layer_weights)
        
        self.epsilon = eps0
        self.eps_decay = eps_decay
        self.eps_freq = eps_decay_freq
        self.min_eps = min_eps
        
        self.gamma = gamma
        
        self.logger = TrainingLog()
        self.testing_freq = testing_freq
        self.n_test = n_test

        self.save_freq = save_freq
        self.save_name = sname
        
        
    def make_model(self, filters, sizes):
        inputs = keras.Input(shape = self.env_shape)

        conv1 = Conv2D(filters[0], sizes[0], padding = 'same', activation = 'relu', input_shape = self.env_shape, data_format = 'channels_last')(inputs)
        conv1 = Conv2D(filters[0], sizes[0], padding = 'same', activation = 'relu')(conv1)
        conv1 = GaussianNoise(0.2)(conv1)
        
        conv2 = Conv2D(filters[1], sizes[1],  padding = 'same', activation = 'relu')(conv1)
        conv2 = GaussianNoise(0.2)(conv2)

        conv3 = Conv2D(filters[2], sizes[2], padding = 'same', activation = 'relu')(conv2)
        conv3 = GaussianNoise(0.2)(conv3)

        conv4 = Conv2D(3, 1, padding = 'same', activation = 'relu')(conv3)
        output = Conv2D(1, 3, padding = 'same', activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1))(conv4)

        model = keras.Model(inputs = inputs, outputs = output)
        model.compile('adam', loss='mse')
        
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
            if steps == 0:
                action = (np.random.randint(self.env.n_rows), np.random.randint(self.env.n_cols))
            else:
                action = self.choose_action(current_state, use_eps = use_epsilon)
            new_state, reward, done, _ = self.env.step(action)
            steps += 1

            if steps > self.env.n_rows * self.env.n_cols:
                done = True

            if log:
                self.logger.log_step(current_state, action, reward, done)

            if done: 
                if log:
                    self.logger.log_final(new_state, steps)
                    return 
                else:
                    # if not logging, send boolean indicator of success
                    if reward == self.env.completion_reward:
                        success = True
                    else:
                        success = False
                    return success

            current_state = new_state

    def run_test(self, n):
        successes = 0
        for _ in range(n):
            success = self.run_episode(log = False, use_epsilon = False)
            successes += success
        return successes/n

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
        vals = rewards + adj * self.gamma # scale by gamma

        for i in range(batch_size):
            x, y = actions[i]

            preds[i, x, y, 0] = vals[i]

        self.model.fit(states, preds, verbose = False)

    def decay_epsilon(self):
        if self.epsilon > self.min_eps:
            self.epsilon *= self.eps_decay

    def save_model(self, f_annot = ''):
        ts = datetime.now().strftime('%y%b%d_%H_%M')
        self.model.save(f'Models/{self.save_name}_{ts}_{f_annot}')

    def training_loop(self, n_eps, batch_size):
        for i in tqdm(range(n_eps)):
            self.run_episode()
            self.fit(batch_size)

            if (i + 1) % self.eps_freq == 0:
                self.decay_epsilon()
            
            if (i + 1) % self.testing_freq == 0:
                r = self.run_test(self.n_test)
                self.logger.tests.append(r)

            if (i + 1) % self.save_freq == 0:
                self.save_model(f_annot = i + 1)

    def show_single(self, fname, seed = None):
        np.random.seed(seed)
        self.env.prime()

        current_state = self.env.network_obs
        steps = 0

        states = []
        vals = []

        while True:
            pred = self.model.predict(current_state).reshape(self.env_shape[0], self.env_shape[1])
            row, col = np.where(pred == pred.max())
            row, col = row[0], col[0]

            states.append(self.env.last_obs)
            vals.append(pred)
            new_state, _, done, _ = self.env.step((row, col))
            steps += 1

            if done:
                last_val = self.model.predict(new_state).reshape(self.env_shape[0], self.env_shape[1])
                vals.append(last_val)
                break

            if steps > 40:
                break
            current_state = new_state

        plt.figure(figsize = (15, 5))

        fig = plt.gcf()
        mi_ax = plt.subplot(1, 3, 1)
        ax1 = plt.subplot(1, 3, 2)
        ax2 = plt.subplot(1, 3, 3)
        
        sns.heatmap(self.env.is_mine, ax = mi_ax)

        def update(i):
            ax1.cla()
            ax2.cla()
            ax1.set_title(f'Step {i}')
            sns.heatmap(states[i], ax = ax1, annot = True, cbar = False, vmin = -1, vmax = 5)
            sns.heatmap(vals[i], ax = ax2, cbar = False)
            
        ani = FuncAnimation(fig, update, steps, interval = 500, repeat_delay = 2000)

        ani.save(f'Images/{fname}.gif', writer = 'imagemagick')





