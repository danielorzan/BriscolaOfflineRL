from utils import one_hot_encoding
from copy import deepcopy
import torch
import numpy as np
import os
import csv

class CardGameEnvironment:
    def __init__(self):
        self.deck_size = 40
        self.cards_left = [i for i in range(self.deck_size)]
        self.end = False
        self.device = 'cpu'

        self.points = {} # dictionary with point value cards
        for i in range(0,self.deck_size,10):
            self.points[i] = 11
            self.points[i+1] = 0
            self.points[i+2] = 10
            self.points[i+3] = 0
            self.points[i+4] = 0
            self.points[i+5] = 0
            self.points[i+6] = 0
            self.points[i+7] = 2
            self.points[i+8] = 3
            self.points[i+9] = 4

        self.max_test_ep_len = 21
        self.eval_batch_size = 1  # required for forward pass
        
        self.state_dim = 40
        self.rtg_scale = 120

    def giveCard(self):
        if len(self.cards_left) == 0:
            self.end = True
            return self.state[-1]
        else:
            idx = list(range(len(self.cards_left)))
            draw = np.random.choice(idx)
            new_card = self.cards_left[draw]
            del self.cards_left[draw]
            return new_card
        
    def playerNxtState(self, action):
        if not self.end:
            new_card = self.giveCard()
        else:
            new_card = 40
        idx = self.state[0].index(action)
        self.state[0][idx] = new_card
        return
    
    def oppNxtState(self):
        if not self.end:
            new_card = self.giveCard()
        else:
            new_card = 40
        idx = self.opp.index(self.opp_card)
        self.opp[idx] = new_card
        return
    
    def count_points(self, card1, card2):
        return self.points[card1] + self.points[card2]
    
    def _givePoints(self, card):
        briscola = self.state[-1]
        points = self.count_points(card, self.opp_card)
        self.current_points = points
        idx = [0,2,9,8,7,6,5,4,3,1] # card's order by value
        card_i = idx.index(card%10)
        opp_i = idx.index(self.opp_card%10)
        if self.opp_card//10 == briscola//10: # checking if opponent's card is briscola
            if card//10 == self.opp_card//10: # checking if agent's card is briscola
                if card_i < opp_i: # win
                    self.points_count[0] += points
                    return True, points
                else: # lose
                    self.points_count[1] += points
                    return False, 0
            else: # lose
                self.points_count[1] += points
                return False, 0
        else:
            if card//10 == briscola//10: # win
                self.points_count[0] += points
                return True, points
            if card//10 == self.opp_card//10:
                if card_i < opp_i: # win
                    self.points_count[0] += points
                    return True, points
                else: # lose
                    self.points_count[1] += points
                    return False, 0
            else:
                if self.first: # win
                    self.points_count[0] += points
                    return True, points
                else: # lose
                    self.points_count[1] += points
                    return False, 0
    
    def reset(self):
        self.game = []
        self.full_game = []
        self.state = [[40,40,40],40,40,40] # initial empty state
        self.cards_left = [i for i in range(self.deck_size)]
        self.end = False
        self.points_count = [0,0] # agent and opponent
        self.current_points = 0

        # initialize new game
        briscola = np.random.choice(self.cards_left)
        self.state[-1] = briscola
        del self.cards_left[briscola]

        # deal cards
        self.state[0] = [self.giveCard() for _ in range(3)]
        self.opp = [self.giveCard() for _ in range(3)]
        self.opp_card = 40

        self.full_game.append([deepcopy(self.state[0]),deepcopy(self.opp)])

        self.first = np.random.randint(0, 2)
                
        # same as timesteps used for training the transformer
        timesteps = torch.arange(start=0, end=self.max_test_ep_len, step=1)
        self.timesteps = timesteps.repeat(self.eval_batch_size, 1).to(self.device)

        with torch.no_grad():

            # zeros place holders
            self.actions = torch.zeros((self.eval_batch_size, self.max_test_ep_len),
                                dtype=torch.int64, device=self.device)

            self.states = torch.zeros((self.eval_batch_size, self.max_test_ep_len, 4, self.state_dim),
                                dtype=torch.float32, device=self.device)

            self.rewards_to_go = torch.zeros((self.eval_batch_size, self.max_test_ep_len, 1),
                                dtype=torch.float32, device=self.device)
            
        return self.state

    def state_encoding(self):
        running_state = []
        for i in range(4):
            running_state.append(one_hot_encoding(self.state[i]))

        self.game.append(deepcopy(self.state))
        self.full_game.append([deepcopy(self.state[0]),deepcopy(self.opp)])
        
        return running_state

    def choose_action(self, model, t, running_state, running_rtg):
            
        # add state in placeholder
        self.states[0,t,:,:] = torch.tensor(running_state, dtype=torch.float32).to(self.device)
        self.rewards_to_go[0, t] = running_rtg

        _, act_preds, _, _ = model.forward(self.timesteps[:,:t+1],
                                    self.states[:,:t+1,:,:],
                                    self.actions[:,:t+1],
                                    self.rewards_to_go[:,:t+1])

        act = act_preds[0,-1].detach()
        action = np.argmax(act.cpu().numpy()) # max out of 41 long array
        self.actions[0, t] = action

        info = 0 # counting unfeasible actions
        self.game.append(action)
        if action not in self.state[0] or action == 40:
            tmp = self.state[0]
            a_tmp = [tmp[i] for i in range(3) if tmp[i] != 40]
            if len(a_tmp) > 0:
                action = np.random.choice(a_tmp)
            else:
                return self.state, 0, info
            self.game.append('not possible action')
            info = 1
            self.game.append(action)

        return action, info

    def step(self, action): # count points   
        self.first, reward = self._givePoints(action)
        self.game.append(reward)

        self.state[2] = deepcopy(self.opp_card)
        if self.first:
            self.playerNxtState(action)
            self.oppNxtState()
        else:
            self.oppNxtState()
            self.playerNxtState(action)

        return reward

    def get_state(self):
        return self.state[0], self.opp, self.points_count

    def render(self):
        print('Agent current points:', self.points_count[0])
        print('Your current points:', self.points_count[1])
        print('Agent current cards:', self.state[0])
        print('Your current cards:', self.opp)
        return
    
    def save_game(self):
        log_dir = "./played_games/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        filename = 'game_points.csv'
        file_path = os.path.join(log_dir, filename)
        file_exists = os.path.isfile(file_path)
        
        # Open the file in append mode
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # If the file doesn't exist, write the header
            if not file_exists:
                writer.writerow(['Agent', 'Human'])  # Change column names as needed

            # Write the data
            writer.writerow(self.points_count)

        return