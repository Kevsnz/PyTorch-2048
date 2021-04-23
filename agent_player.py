import torch
import random
import numpy as np
from agent_net import AgentNet
from game import Game2048

class AgentPlayer:
    TURN_LIMIT = 2 ** Game2048.TARGET_SCORE
    MAX_SEQ_INVALID = 4

    def __init__(self, net: AgentNet, game: Game2048):
        self.net = net
        self.game = game
        self.invalidSeq = 0
    
    
    def playEpisode(self, invalidEnd = False):
        self.game.reset()
        self.invalidSeq = 0

        states = []
        actions = []
        rewards = []
        endeds = []
        newStates = []
        while True:
            s, a, r, e, s1 = self._makeTurn(self.net, self.game, invalidEnd)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            endeds.append(e)
            newStates.append(s1)

            if e:
                break
            
        return states, actions, rewards, endeds, newStates
    

    def makeTurn(self, invalidEnd = False):
        return self._makeTurn(self.net, self.game, invalidEnd)
    

    def _makeTurn(self, net: AgentNet, game: Game2048, invalidEnd = False):
        state = game.board.copy()
        action = net(net.prepareInputs(state))[0].squeeze(0)
        action = torch.softmax(action, dim=0).detach().numpy()

        dir = np.random.choice(len(action), size=1, p=action)
        reward, ended, valid = game.swipe(dir)
        if valid:
            self.invalidSeq = 0
        else:
            reward = -20
            self.invalidSeq += 1
            ended = invalidEnd
            if self.invalidSeq == self.MAX_SEQ_INVALID:
                ended = True
                self.invalidSeq = 0
        
        if game.swipeCount >= self.TURN_LIMIT:
            ended = True
        
        return state, dir, reward, ended, game.board.copy()

