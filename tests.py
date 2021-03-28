from agent_net import AgentNet
from game import Game2048
import random
import torch
import numpy as np

def TestAgentNet():
    print('Running AgentNet tests...')
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
            #print(f'PASS: Coords ({i}, {j}) value {val}')
            passes += 1
    return passes, fails


def TestGame():
    print('Running Game2048 tests...')
    passes = 0
    fails = 0

    cases = [
        {
            'board':[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 0,
            'ended': False
        },
        {
            'board':[
                [0, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 0,
            'ended': False
        },
        {
            'board':[
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 2,
            'ended': False
        },
        {
            'board':[
                [0, 0, 0, 0],
                [0, 1, 1, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 5,
            'ended': False
        },
        {
            'board':[
                [0, 0, 0, 0],
                [1, 0, 1, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 5,
            'ended': False
        },
        {
            'board':[
                [0, 0, 0, 0],
                [1, 0, 1, 2],
                [2, 0, 2, 3],
                [0, 0, 0, 0]],
            'score': 12,
            'ended': False
        },
        {
            'board':[
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]],
            'score': 0,
            'ended': True
        },
        {
            'board':[
                [0, 0, 0, 0],
                [0,10, 0,10],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 11,
            'ended': True
        },
    ]

    game = Game2048()

    for i in range(len(cases)):
        case = cases[i]
        game.reset()
        game.board = np.array(case['board'])
        score, ended = game.swipe(2)
        if score != case['score'] or ended != case['ended']:
            print(f'FAIL: Test {i}, got: {score} ({ended}), expected {case["score"]} ({case["ended"]})')
            print(game.boardAsString())
            fails += 1
        else:
            passes += 1

    return passes, fails


if __name__ == '__main__':
    print('Running tests...')

    p, f = TestAgentNet()
    print(f'TestAgentNet: {p} Passed, {f} failed')

    p, f = TestGame()
    print(f'TestGame: {p} Passed, {f} failed')

    print('Done!')
