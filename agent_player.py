from agent_net import AgentNet
from game import Game2048
import torch
import torch.functional as F
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
        while True:
            s, a, r, e = self.makeTurn(self.net, self.game, eps)

            if e:
                r += self.game.score

            states.append(s)
            actions.append(a)
            rewards.append(r)
            endeds.append(e)

            if e:
                break
            
        return states, actions, rewards, endeds
    

    def makeTurn(self, net: AgentNet, game: Game2048, eps: float):
        state = net.prepareInput(game.board)
        action = net(state)

        if np.random.rand() < eps:
            dir = np.random.randint(4)
        else:
            actionProb = torch.softmax(action, dim=0)
            dir = np.random.choice(range(0, 4), 1, p=actionProb.cpu().detach().numpy())
        
        reward, ended = game.swipe(dir)
        return state, action, reward, ended

