import numpy as np
import collections

class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()


    def clear(self):
        self.stateBuffer = collections.deque(maxlen=self.capacity)
        self.actionBuffer = collections.deque(maxlen=self.capacity)
        self.rewardBuffer = collections.deque(maxlen=self.capacity)
        self.termBuffer = collections.deque(maxlen=self.capacity)
        self.newStateBuffer = collections.deque(maxlen=self.capacity)


    def addRange(self, states, actions, rewards, terms, newStates):
        self.stateBuffer.extend(states)
        self.actionBuffer.extend(actions)
        self.rewardBuffer.extend(rewards)
        self.termBuffer.extend(terms)
        self.newStateBuffer.extend(newStates)


    def add(self, state, action, reward, terminal, newState):
        self.stateBuffer.append(state)
        self.actionBuffer.append(action)
        self.rewardBuffer.append(reward)
        self.termBuffer.append(terminal)
        self.newStateBuffer.append(newState)


    def count(self):
        return len(self.stateBuffer)


    def sample(self, count):
        idxs = np.random.choice(len(self.stateBuffer), size=count, replace=False)

        states, actions, rewards, terms, newStates = zip(*[(self.stateBuffer[i], self.actionBuffer[i], self.rewardBuffer[i], self.termBuffer[i], self.newStateBuffer[i]) for i in idxs])
        
        return np.array(states), \
            np.array(actions, dtype=np.int64), \
            np.array(rewards, dtype=np.float32), \
            np.array(terms, dtype=np.bool8), \
            np.array(newStates)
