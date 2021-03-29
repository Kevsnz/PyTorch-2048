import numpy as np

class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()


    def clear(self):
        self.stateBuffer = []
        self.actionBuffer = []
        self.rewardBuffer = []
        self.termBuffer = []
        self.newStateBuffer = []


    def add(self, states, actions, rewards, terms, newStates):
        self.stateBuffer.extend(states)
        self.actionBuffer.extend(actions)
        self.rewardBuffer.extend(rewards)
        self.termBuffer.extend(terms)
        self.newStateBuffer.extend(newStates)

        if len(self.stateBuffer) > self.capacity:
            self.stateBuffer = self.stateBuffer[-self.capacity:]
            self.actionBuffer = self.actionBuffer[-self.capacity:]
            self.rewardBuffer = self.rewardBuffer[-self.capacity:]
            self.termBuffer = self.termBuffer[-self.capacity:]
            self.newStateBuffer = self.newStateBuffer[-self.capacity:]


    def sample(self, count):
        idxs = np.random.choice(len(self.stateBuffer), size=count, replace=False)

        # states = self.stateBuffer[idxs]
        # actions = self.actionBuffer[idxs]
        # rewards = self.rewardBuffer[idxs]
        # terms = self.termBuffer[idxs]
        # newStates = self.newStateBuffer[idxs]

        states, actions, rewards, terms, newStates = zip(*
            [(self.stateBuffer[i], self.actionBuffer[i], self.rewardBuffer[i], self.termBuffer[i], self.newStateBuffer[i]) for i in idxs])
        
        return np.array(states), \
            np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(terms, dtype=np.uint8), \
            np.array(newStates)
