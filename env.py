import numpy as np
from copy import deepcopy
import torch
from src.agents.NNAgent import NNAgent
from src.envs.two_player_briscola.BriscolaConstants import Constants

class BriscolaEnv:

    def __init__(self, rand=False):
        self.deck_size = 40
        self.cards_left = [i for i in range(self.deck_size)]
        self.end = False
        self.rand = rand

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

        self.state = [[40,40,40],40,40,40] # initial empty state
        self.opp = [40,40,40]
        self.opp_card = 40
        self.first = 0 # turn
        self.game = []
        self.thrown_cards = []
        self.points_count = [0,0]

        # Lorenzo's agent
        device = "cpu"
        observation_shape = (162,)
        action_size = 40
        self.trained_previous = NNAgent(observation_shape, action_size, hidden_size=256).to(device)
        self.trained_previous.load_state_dict(torch.load("briscola-bot-v3.pt"))
        self.lore_info = 0 # counting number of unfeasible actions

        self.wrong_games = []

    def observe(self, action=40):
        observation = torch.zeros((4, Constants.deck_cards), dtype=torch.float32)
        observation[0, self.thrown_cards] = 1 # thrown cards during the game
        observation[1, self.state[-1]] = 1 # briscola
        if self.first: # lorenzo plays second
            observation[2, action] = 1
        for card in self.opp:
            if card != 40:
                observation[3, card] = 1
        observation = torch.cat((
            observation.view(-1),
            torch.tensor([
                self.points_count[1] / Constants.total_points,
                self.points_count[0] / Constants.total_points
            ])
        ), dim=0)

        action_mask = torch.zeros((Constants.deck_cards,), dtype=torch.int64)
        for card in self.opp:
            if card != 40:
                action_mask[card] = 1

        return observation, action_mask
    
    def lorenzo_action(self, action=40):
        observations, action_masks = self.observe(action)
        l_action = self.trained_previous.get_actions(observations, action_masks)
        info = 0 # counting unfeasible actions
        if l_action not in self.opp:
            tmp = self.opp
            a_tmp = [tmp[i] for i in range(3) if tmp[i] != 40]
            if len(a_tmp) > 0:
                l_action = np.random.choice(a_tmp)
            else:
                return 40 # finished cards in hand
            self.game.append('lorenzo not possible action')
            info = 1
        self.opp_card = l_action.item()
        return info
    
    def rand_action(self):
        tmp = self.opp
        a_tmp = [tmp[i] for i in range(3) if tmp[i] != 40]
        if len(a_tmp) > 0:
            self.opp_card = np.random.choice(a_tmp)
            return
        else: # finished cards in hand
            self.opp_card = 40
            return

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

    def count_points(self, card1, card2):
        return self.points[card1] + self.points[card2]

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

    def _givePoints(self, card):
        briscola = self.state[-1]
        points = self.count_points(card, self.opp_card)
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
        self.state = [[40,40,40],40,40,40] # initial empty state
        self.cards_left = [i for i in range(self.deck_size)]
        self.end = False
        self.thrown_cards = []
        self.points_count = [0,0]

        # initialize new game
        briscola = np.random.choice(self.cards_left)
        self.state[-1] = briscola
        del self.cards_left[briscola]

        # deal cards
        self.state[0] = [self.giveCard() for _ in range(3)]
        self.opp = [self.giveCard() for _ in range(3)]
        self.opp_card = 40

        self.first = np.random.randint(0, 2)
        if not self.first: # play second
            if self.rand:
                self.rand_action()
            else:
                self.lore_info += self.lorenzo_action()
            self.state[1] = deepcopy(self.opp_card)
            self.thrown_cards.append(deepcopy(self.opp_card))

        self.game.append(deepcopy(self.state))

        return self.state

    def step(self, action):
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
        self.thrown_cards.append(action)

        if self.first: # playing first, need opponent's action
            if self.rand:
                self.rand_action()
            else:
                self.lore_info += self.lorenzo_action(action=action)
            self.thrown_cards.append(deepcopy(self.opp_card))

        self.first, reward = self._givePoints(action)
        self.game.append(reward)

        self.state[2] = deepcopy(self.opp_card)
        self.playerNxtState(action)
        self.oppNxtState()
        
        if any(element != 40 for element in self.opp):
            if not self.first: # play second
                if self.rand:
                    self.rand_action()
                else:
                    self.lore_info += self.lorenzo_action()
                self.state[1] = deepcopy(self.opp_card)
                self.thrown_cards.append(deepcopy(self.opp_card))
            else: # play first
                self.state[1] = 40
        else:
            self.state[1] = 40

        self.game.append(deepcopy(self.state))
        return self.state, reward, info