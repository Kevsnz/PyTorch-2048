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
        
        try:
            if len(self.states) == self.stepCount:
                # return sr, ar, rr, tr, s1
                return self.states[0], self.actions[0], self.rewards[0], self.terminals[0], s1
            
            return None, None, None, None, None
        finally:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.terminals.append(t)

        