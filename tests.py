import random
import torch
import numpy as np
from agent_net import AgentNet
from game import Game2048
from experience_unroller import ExperienceUnroller

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
        res = agent.prepareInputs(np.array(board))
        if res[0, i, j, val] != 1.0:
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
        { # 0
            'board':[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 0,
            'ended': False,
            'valid': False
        },
        { # 1
            'board':[
                [0, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 0,
            'ended': False,
            'valid': True
        },
        { # 2
            'board':[
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 4,
            'ended': False,
            'valid': True
        },
        { # 3
            'board':[
                [0, 0, 0, 0],
                [0, 1, 1, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 4,
            'ended': False,
            'valid': True
        },
        { # 4
            'board':[
                [0, 0, 0, 0],
                [1, 0, 1, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 4,
            'ended': False,
            'valid': True
        },
        { # 5
            'board':[
                [0, 0, 0, 0],
                [1, 0, 1, 2],
                [2, 0, 2, 3],
                [0, 0, 0, 0]],
            'score': 12,
            'ended': False,
            'valid': True
        },
        { # 6
            'board':[
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 4,
            'ended': False,
            'valid': True
        },
        { # 7
            'board':[
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]],
            'score': 0,
            'ended': False,
            'valid': False
        },
        { # 8
            'board':[
                [0, 0, 0, 0],
                [0,10, 0,10],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 2048,
            'ended': True,
            'valid': True
        },
        { # 9
            'board':[
                [4, 4, 3, 4],
                [4, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7]],
            'score': -11,
            'ended': True,
            'valid': True
        },
        { # 10
            'board':[
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
            'score': 8,
            'ended': False,
            'valid': True
        },
    ]

    game = Game2048()

    for i in range(len(cases)):
        case = cases[i]
        game.reset()
        game.board = np.array(case['board'])
        score, ended, valid = game.swipe(2)
        if score != case['score'] or ended != case['ended'] or valid != case['valid']:
            print(f'FAIL: Test {i}, got: {score} ({ended}, {valid}), expected {case["score"]} ({case["ended"]}, {case["valid"]})')
            print(game.boardAsString())
            fails += 1
        else:
            passes += 1

    return passes, fails


def TestUnroller():
    print('Running Experience Unroller tests...')

    testCases = [
        { # 0
            'sc': 0,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4]
            ],
            'outp': [1, 2, 3, False, 4]
        },
        { # 1
            'sc': 1,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4]
            ],
            'outp': [None, None, None, None, 4]
        },
        { # 2
            'sc': 1,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4],
                [5, 6, 7, False, 8]
            ],
            'outp': [1, 2, 6.5, False, 8]
        },
        { # 3
            'sc': 1,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, True, 4],
                [5, 6, 7, False, 8]
            ],
            'outp': [1, 2, 3, True, 8]
        },
        { # 4
            'sc': 1,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4],
                [5, 6, 7, True, 8]
            ],
            'outp': [1, 2, 6.5, True, 8]
        },
        { # 5
            'sc': 2,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4],
                [5, 6, 7, False, 8],
                [9,10,11, False,12]
            ],
            'outp': [1, 2, 9.25, False, 12]
        },
        { # 6
            'sc': 2,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4],
                [5, 6, 7, False, 8],
                [9,10,11, False,12],
                [13,14,15, False,16]
            ],
            'outp': [5, 6, 16.25, False, 16]
        },
        { # 7
            'sc': 2,
            'gamma': 0.5,
            'inp': [
                [1, 2, 3, False, 4],
                [5, 6, 7, False, 8],
                [9,10,11,  True,12],
                [13,14,15, False,16]
            ],
            'outp': [5, 6, 12.5, True, 16]
        }
    ]

    passes = fails = 0

    for i in range(len(testCases)):
        tc = testCases[i]
        exp = ExperienceUnroller(tc['sc'], tc['gamma'])
        for inp in tc['inp']:
            s,a,r,t,s1 = exp.add(inp[0], inp[1], inp[2], inp[3], inp[4])
        if not(s == tc['outp'][0] and a == tc['outp'][1] and r == tc['outp'][2] and t == tc['outp'][3] and s1 == tc['outp'][4]):
            print(f'#{i} FAIL: Step count {tc["sc"]}: expected ({tc["outp"][0]}, {tc["outp"][1]}, {tc["outp"][2]}, {tc["outp"][3]}, {tc["outp"][4]}), got ({s}, {a}, {r}, {t}, {s1})')
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

    p, f = TestUnroller()
    print(f'TestUnroller: {p} Passed, {f} failed')

    print('Done!')
