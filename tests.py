from agent_net import AgentNet
import random
import torch

def TestAgentNet():
    agent = AgentNet()
    passes = 0
    fails = 0

    for i in range(20):
        board = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
        i = random.randint(0,3)
        j = random.randint(0,3)
        val = random.randint(1,11)
        board[i][j] = val
        res = agent.prepareInput(board)
        if res[i,j,val] != 1.0:
            print(f'FAIL: Coords ({i}, {j}) value {val}')
            print(res)
            fails += 1
        else:
            print(f'PASS: Coords ({i}, {j}) value {val}')
            passes += 1
    return passes, fails


if __name__ == '__main__':
    print('Running tests...')

    p, f = TestAgentNet()
    print(f'TestAgentNet: {p} Passed, {f} failed')

    print('Done!')
