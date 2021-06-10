import gym
from gym.spaces import Tuple, Discrete
import numpy as np



class Minefield(gym.Env):
    '''
    Class representing the game board. 
    
    Internal variables: 
    
    action_space: set of possible actions, each a tuple of ints representing (row, column) coordinates of click
    n_rows, n_cols: size of board
    n_mines: count of mines on board
    
    safe_reward: reward for clicking a square with no mine
    repeat_loss: loss for clicking an already revealed square
    mine_loss: loss for clicking on a mine
    completion_reward: reward for clicking every non-mine square
    
    coords: coordinates of mines, tuples of ints
    is_mine: array showing mine locations. 1 or 0, n_row x n_col
    revealed: has square been shown to player? boolean
    '''
    
    def __init__(self, nrows, ncols, nmines, safe_reward = 1, repeat_loss = -1, mine_loss = -10, completion_reward = 20):
        super().__init__()
        
        self.action_space = Tuple((Discrete(nrows), Discrete(ncols)))
        self.n_rows = nrows
        self.n_cols = ncols
        self.n_mines = nmines
        
        self.safe_reward = safe_reward
        self.repeat_loss = repeat_loss
        self.mine_loss = mine_loss
        self.completion_reward = completion_reward
        
        self.prime()
        
    def prime(self):
        '''
        Method for before starting a game, just clears last obs and indicates no activate game
        '''
        
        self.started = False
        obs = np.ones((self.n_rows, self.n_cols)) * -1
        self.last_obs = obs
        
    
    def start_game(self, coord):
        '''
        Takes first click and begins a game. First click must be on a square with 0 neighboring mines. 
        Essentially just keeps initalizing board until no neighbors have mines. 
        Could change to do a single initialization, choosing from non-neighbor squares to place all mines. 
        '''
        self.started = True
        
        mine_locs = np.random.choice(self.n_rows * self.n_cols, size = self.n_mines, replace = False)
        mine_coords = [(x % self.n_rows, x // self.n_rows) for x in mine_locs]
        is_mine = np.zeros((self.n_rows, self.n_cols))
        
        for pair in mine_coords:
            is_mine[pair] = 1
        
        left, right = max(coord[0] - 1, 0), min(coord[0] + 2, self.n_rows)
        top, bottom = max(coord[1] - 1, 0), min(coord[1] + 2, self.n_cols)
        while is_mine[left:right, top:bottom].sum() > 0: # more than 0 mines in vacinity of click
            mine_locs = np.random.choice(self.n_rows * self.n_cols, size = self.n_mines, replace = False)
            mine_coords = [(x % self.n_rows, x // self.n_rows) for x in mine_locs]
            is_mine = np.zeros((self.n_rows, self.n_cols))

            for pair in mine_coords:
                is_mine[pair] = 1
            
        self.coords = mine_coords
        self.is_mine = is_mine
        
        self.get_neighbor_counts()
        self.revealed = np.zeros((self.n_rows, self.n_cols))
        
    def get_neighbor_counts(self):  
        '''
        Gives number of neighboring mines at every square. Mines are just -1. 
        Functionally just a 3x3 convolution of 1s over the is_mine array. 
        Could do a pointwise operation on each point with similar logic to start_game
        '''
        counts = np.zeros((self.n_rows + 2, self.n_cols + 2))
        
        for x, y in self.coords:
            for i in range(x, x + 3):
                for j in range(y, y + 3):
                    if (i - 1, j - 1) in self.coords:
                        counts[i, j] = -1
                    else: 
                        counts[i, j] += 1
        
        counts = counts[1:self.n_rows + 1, 1: self.n_cols + 1]
        
        
        self.counts = counts
        
    def explode_out(self, array):
        '''
        Formats the input (a 2D array of [-1, 8] integers) as a 4D array as expected by the neural network. 
        Each slice along the last dimension represents a given value in for each cell. 
        First slice is 0/1 shown or not. Second is shown and contains 0, third is shown and contains 1, etc.         
        '''
        out_array = np.zeros((1, array.shape[0], array.shape[1], 10))

        for x, row in enumerate(array):
            for y, el in enumerate(row):
                out_array[0, x, y, int(el + 1)] = 1

        return np.concatenate([out_array, np.ones((1, 8, 8, 1))], axis = -1)
        
    def zero_block(self, coord, ret = None):
        '''
        Block to be revealed based on a click. If a cell contains a non-zero number, reveal only that cell. 
        Otherwise, find the contiguous block of 0s and reveal that.         
        '''
        if ret is None:
            ret = set()
            return_ret = True
        else:
            return_ret = False

        ret.add(coord)

        if self.counts[coord] == 0:
            is_zero = True
        else:
            is_zero = False
            if return_ret:
              return ret
            else:
              return

        up = max(0, coord[0] - 1)
        down = min(self.n_rows - 1, coord[0] + 1)

        l = max(0, coord[1] - 1)
        r = min(self.n_cols - 1, coord[1] + 1)

        for i in range(up, down + 1):
            for j in range(l, r + 1):
                if self.counts[i, j] == 0:
                    if (i, j) not in ret:
                        self.zero_block((i,j), ret)
                elif is_zero:
                    ret.add((i, j))

        if return_ret:
            return ret
        
    @property
    def inherent_value(self):
        '''
        "Actual" q value of each cell, unknown to the learner. All mines cause self.mine_loss, repeats cause self.repeat_loss, 
        others cause self.safe_reward. 
        '''
        mine_losses = self.is_mine * self.mine_loss
        repeat_losses = self.revealed * self.repeat_loss
        reveal_gains = (1 - self.revealed) * self.safe_reward
        
        non_mines = (repeat_losses + reveal_gains) * (1 - self.is_mine)
        
        total_value = non_mines + mine_losses
        
        return total_value
        
    def step(self, coord):
      
        done = False
        if not self.started:
            self.start_game(coord)
            
        if self.is_mine[coord]: #Losing condition
            reward = self.mine_loss
            to_reveal = set(coord)
            done = True
        else: 
            if self.revealed[coord]: # Already visible square, so repeat loss and don't show anything new
                reward = self.repeat_loss
                to_reveal = set() 
            else: # Show new squares
                reward = self.safe_reward
                to_reveal = self.zero_block(coord)
                
        for pair in to_reveal: # update to show what is revealed
            self.revealed[pair] = 1
        
        n = self.n_rows * self.n_cols
        if self.revealed.sum() == (n - self.n_mines): # winning condition, if last click was not a mine
            if not done: 
                reward = self.completion_reward
                done = True
                #update the rewards matrix
                #reward matrix could be: self.inherent_value * -ones(to_reveal) + self.completion_reward * ones(to_reveal)
            
        
        obs = self.revealed * self.counts + (1 - self.revealed) * -1
        self.last_obs = obs
        
        # prepare for reset
        if done:
            self.started = False
        
        return self.network_obs, reward, done, None
      
    @property
    def network_obs(self):
        return self.explode_out(self.last_obs)
    
    def render(self):
        print(self.revealed * self.counts + (1 - self.revealed) * -1)

