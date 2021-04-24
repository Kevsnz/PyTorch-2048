from collections import deque

class ExperienceUnroller:
    def __init__(self, stepCount = 1, gamma = 0.99):
        assert(stepCount >= 0)
        self.stepCount = stepCount
        self.gamma = gamma

        if stepCount > 0:
            self.states = deque(maxlen=stepCount)
            self.actions = deque(maxlen=stepCount)
            self.rewards = deque(maxlen=stepCount)
            self.terminals = deque(maxlen=stepCount)
        pass


    def add(self, s, a, r, t, s1):
        if self.stepCount == 0:
            return s, a, r, t, s1

        for i in range(1, len(self.rewards)+1):
            if self.terminals[-i]:
                break

            self.rewards[-i] += r * self.gamma**i
            self.terminals[-i] = t # or self.terminals[-i]
        
        sr = ar = rr = tr = None
        if len(self.states) == self.stepCount:
            sr = self.states[0]
            ar = self.actions[0]
            rr = self.rewards[0]
            tr = self.terminals[0]
        
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.terminals.append(t)
        return sr, ar, rr, tr, s1


    def clear(self):
        if self.stepCount > 0:
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.terminals.clear()

        