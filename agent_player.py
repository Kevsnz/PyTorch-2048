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
        newStates = []
        while True:
            s, a, r, e, s1 = self.makeTurn(self.net, self.game, eps)

            if e:
                r += self.game.score

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
            action = net(net.prepareInput(state))
            dir = torch.argmax(action).item()
            pass
            #actionProb = torch.softmax(action, dim=0)
            #dir = np.random.choice(range(0, 4), 1, p=actionProb.cpu().detach().numpy())
        
        reward, ended = game.swipe(dir)
        newState = game.board
        return state, dir, reward, ended, newState

