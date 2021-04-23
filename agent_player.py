from agent_net import AgentNet
from game import Game2048
import torch
import random

class AgentPlayer:
    TURN_LIMIT = 2 ** Game2048.TARGET_SCORE
    MAX_SEQ_INVALID = 4

    def __init__(self, net: AgentNet, game: Game2048):
        self.net = net
        self.game = game
        self.invalidSeq = 0
    
    
    def playEpisode(self, invalidEnd = False, eps = 0.1):
        self.game.reset()
        self.invalidSeq = 0

        states = []
        actions = []
        rewards = []
        endeds = []
        newStates = []
        while True:
            s, a, r, e, s1 = self._makeTurn(self.net, self.game, invalidEnd, eps)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            endeds.append(e)
            newStates.append(s1)

            if e:
                break
            
        return states, actions, rewards, endeds, newStates
    

    def makeTurn(self, invalidEnd = False, eps = 0.1):
        return self._makeTurn(self.net, self.game, invalidEnd, eps)
    

    def _makeTurn(self, net: AgentNet, game: Game2048, invalidEnd = False, eps = 0.1):
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
            dir = torch.argmax(action).item()
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

