import torch
import random
import numpy as np
from agent_net import AgentNet
from game import Game2048

class AgentPlayer:
    def __init__(self, net: AgentNet, game: Game2048):
        self.net = net
        self.game = game
    
    
    def playEpisode(self):
        self.game.reset()

        states = []
        actions = []
        rewards = []
        endeds = []
        newStates = []
        while True:
            s, a, r, e, s1 = self._makeTurn(self.net, self.game)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            endeds.append(e)
            newStates.append(s1)

            if e:
                break
            
        return states, actions, rewards, endeds, newStates
    

    def makeTurn(self):
        return self._makeTurn(self.net, self.game)
    

    def _makeTurn(self, net: AgentNet, game: Game2048):
        state = game.board.copy()
        action = net(net.prepareInputs(state))[0].squeeze(0)
        action = torch.softmax(action, dim=0).detach().numpy()

        dir = np.random.choice(len(action), size=1, p=action)
        reward, ended, valid = game.swipe(dir)
        if not valid:
            reward = -20
            ended = True
        
        return state, dir, reward, ended, game.board.copy()

