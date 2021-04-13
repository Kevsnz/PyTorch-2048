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


    def addRange(self, states, actions, rewards, terms, newStates):
        max = 1.0 if len(self.stateBuffer) == 0 else self.priorities.max()
        for _ in range(len(states)):
            self.priorities[self.pos] = max
            self.pos = (self.pos + 1) % self.capacity
        
        self.stateBuffer.extend(states)
        self.actionBuffer.extend(actions)
        self.rewardBuffer.extend(rewards)
        self.termBuffer.extend(terms)
        self.newStateBuffer.extend(newStates)


    def add(self, state, action, reward, terminal, newState):
        self.priorities[self.pos] = 1.0 if len(self.stateBuffer) == 0 else self.priorities.max()
        self.pos = (self.pos + 1) % self.capacity
        self.stateBuffer.append(state)
        self.actionBuffer.append(action)
        self.rewardBuffer.append(reward)
        self.termBuffer.append(terminal)
        self.newStateBuffer.append(newState)


    def count(self):
        return len(self.stateBuffer)


    def sample(self, count, beta = 0.4):
        ps = self.priorities if len(self.stateBuffer) == self.capacity else self.priorities[:self.pos]
        ps = ps ** self.alpha
        ps /= ps.sum()

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
        for i, p in zip(idxs, prios):
            self.priorities[i] = p
