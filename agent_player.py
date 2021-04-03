from agent_net import AgentNet
from game import Game2048
import torch
import random

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
        state = game.board.copy()

        if random.random() < eps:
            valid = False
            moves = [0, 1, 2, 3]
            while not valid:
                dir = moves[random.randrange(len(moves))]
                reward, ended, valid = game.swipe(dir)
                moves.remove(dir)
        else:
            action = net(net.prepareInputs(state)).squeeze(0)
            valid = False
            while not valid:
                dir = torch.argmax(action).item()
                reward, ended, valid = game.swipe(dir)
                action[dir] = torch.min(action) - 0.1
        
        return state, dir, reward, ended, game.board.copy()

