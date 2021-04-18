import numpy as np
import collections

class ExperienceBuffer:
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.clear()


    def clear(self):
        self.stateBuffer = collections.deque(maxlen=self.capacity)
        self.actionBuffer = collections.deque(maxlen=self.capacity)
        self.rewardBuffer = collections.deque(maxlen=self.capacity)
        self.termBuffer = collections.deque(maxlen=self.capacity)
        self.newStateBuffer = collections.deque(maxlen=self.capacity)
        self.priorities = np.zeros(self.capacity)
        self.pos = 0
        self.maxPrio = 1.0
        self.sumPrio = 0.0


    def addRange(self, states, actions, rewards, terms, newStates):
        if len(self.stateBuffer) == self.capacity:
            count = len(states)
            if self.pos < self.capacity - count:
                self.maxPrio = max(self.priorities[:self.pos].max(), self.priorities[self.pos + count:].max())
            else:
                self.maxPrio = self.priorities[count - (self.capacity - self.pos):self.pos].max()
        
        for _ in range(len(states)):
            self.priorities[self.pos] = self.maxPrio
            self.pos = (self.pos + 1) % self.capacity
        self.sumPrio = (self.priorities  ** self.alpha).sum()
        
        self.stateBuffer.extend(states)
        self.actionBuffer.extend(actions)
        self.rewardBuffer.extend(rewards)
        self.termBuffer.extend(terms)
        self.newStateBuffer.extend(newStates)


    def add(self, state, action, reward, terminal, newState):
        oldFirstPrio = self.priorities[self.pos]

        if len(self.stateBuffer) == self.capacity and self.maxPrio == oldFirstPrio:
                self.priorities[self.pos] = self.priorities[self.pos-1]
                self.maxPrio = self.priorities.max()

        self.sumPrio += self.maxPrio ** self.alpha - oldFirstPrio ** self.alpha
        
        self.priorities[self.pos] = self.maxPrio
        self.pos = (self.pos + 1) % self.capacity
        self.stateBuffer.append(state)
        self.actionBuffer.append(action)
        self.rewardBuffer.append(reward)
        self.termBuffer.append(terminal)
        self.newStateBuffer.append(newState)


    def count(self):
        return len(self.stateBuffer)


    def sample(self, count, beta = 0.4):
        ps = (self.priorities ** self.alpha) if len(self.stateBuffer) == self.capacity else self.priorities[:self.pos] ** self.alpha
        ps /= self.sumPrio

        idxs = np.random.choice(len(self.stateBuffer), size=count, p=ps)
        if len(self.stateBuffer) == self.capacity:
            idxs -= self.pos

        states, actions, rewards, terms, newStates = zip(*[(self.stateBuffer[i], self.actionBuffer[i], self.rewardBuffer[i], self.termBuffer[i], self.newStateBuffer[i]) for i in idxs])
        ws = (len(self.stateBuffer) * ps[idxs]) ** (-beta)
        ws /= ws.max()
        
        return np.array(states, copy=False), \
            np.array(actions, dtype=np.int64, copy=False), \
            np.array(rewards, dtype=np.float32, copy=False), \
            np.array(terms, dtype=np.bool8, copy=False), \
            np.array(newStates, copy=False), \
            idxs, ws


    def updatePriorities(self, idxs, prios):
        unique = np.unique(idxs)
        self.sumPrio -= (self.priorities[unique] ** self.alpha).sum()

        oldMax = self.priorities[unique].max()
        newMax = prios.max()
        if self.maxPrio < newMax:
            self.maxPrio = newMax
            oldMax = False
        
        self.priorities[idxs] = prios

        if self.maxPrio == oldMax:
            self.maxPrio = self.priorities.max()

        self.sumPrio += (self.priorities[unique] ** self.alpha).sum()
        pass
