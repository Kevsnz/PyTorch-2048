from agent_net import AgentNet
from game import Game2048
import torch
import numpy as np

class AgentPlayer:
    def __init__(self, net: AgentNet, game: Game2048):
        self.net = net
        self.game = game
    
    
    def playEpisode(self, eps = 0.1):
        self.game.reset()

        states = []
        actions = []
        rewards = []
        endeds = []
        newStates = []
        while True:
            s, a, r, e, s1 = self.makeTurn(self.net, self.game, eps)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            endeds.append(e)
            newStates.append(s1)

            if e:
                break
            
        return states, actions, rewards, endeds, newStates
    

    def makeTurn(self, net: AgentNet, game: Game2048, eps: float):
        state = game.board

        if np.random.rand() < eps:
            dir = np.random.randint(4)
        else:
            action = net(net.prepareInputs(state))
            dir = torch.argmax(action).item()
        
        reward, ended, valid = game.swipe(dir)
        newState = game.board
        return state, dir, reward, ended, newState

