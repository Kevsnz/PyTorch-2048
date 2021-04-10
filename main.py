import random
import time
import torch
import datetime
import os
import numpy as np
from game import Game2048
from agent_net import AgentNet
from agent_player import AgentPlayer
from experience_buffer import ExperienceBuffer
from tensorboardX import SummaryWriter

exp_capacity = 10000
initial_exp_gathering = 2000
targetSyncInterval = 3000

epsilon_initial = 1
epsilon_final = 0.02
epsilon_decay_time = 200000
epsilon_decay_amount = epsilon_initial - epsilon_final

batch_size = 16
gamma = 0.99
learningRate = 0.0005

EVAL_GAMES = 10

def silentPlayout(game):
    turn = 0

    while True:
        turn += 1
        dir = random.randrange(4)
        _, ended, _ = game.swipe(dir)
        if ended:
            if game.score == game.target_score:
                return turn, game.score, True
            
            return turn, game.score, False


def playSomeRandomGames(game):
    for _ in range(150):
        game.reset()
        turns, score, won = silentPlayout(game)

        if won:
            print(f'Victory! Target score of {2**score} reached after {turns} turns.')
        else:
            print(f'Game Over! No space for new number after {turns} turns (final score: {2**score}).')


def playAndPrintEpisode(player, eps = 0.1):
    s, a, r, e, _ = player.playEpisode(eps)
    game = player.game

    for i in range(0,len(s)):
        print(f'#{i} State:')
        print(f'{s[i]}')
        print(f'Action: {a[i]} ({game.MOVE_STRING[a[i]]}), Reward: {r[i]}, Ended: {e[i]}')
    
    print(f'Final score: {game.totalScore}')


def playSomeGames(game, net, count):
    agent = AgentPlayer(net, game)

    avgScore = 0
    maxScore = 0
    minScore = game.TARGET_SCORE + 1
    avgTotal = 0
    minTotal = 2**30
    maxTotal = 0
    for i in range(count):
        agent.playEpisode(0)
        avgScore += game.score
        maxScore = max(maxScore, game.score)
        minScore = min(minScore, game.score)
        avgTotal += game.totalScore
        maxTotal = max(maxTotal, game.totalScore)
        minTotal = min(minTotal, game.totalScore)
    
    return minScore, avgScore / count, maxScore, minTotal, avgTotal / count, maxTotal


def playAndLearn(agentNet, targetNet, player):
    expBuffer = ExperienceBuffer(exp_capacity)
    device = AgentNet.device

    writer = SummaryWriter(logdir=os.path.join('tensorboard', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    reportInterval = 2000

    sampleCountTotal = 0
    sampleLastReport = 0
    episodeCountTotal = 0
    lastSyncSample = 0
    timeLastReport = time.perf_counter()
    epsilon = epsilon_initial

    lossFunc = torch.nn.MSELoss()
    optim = torch.optim.Adam(agentNet.parameters(), lr=learningRate)

    lossAcc = 0
    lossCnt = 0
    episodeLengths = [0]

    try:
        while True:
            if sampleCountTotal > initial_exp_gathering:
                part = min(1.0, (sampleCountTotal - initial_exp_gathering) / epsilon_decay_time)
                epsilon = epsilon_initial - epsilon_decay_amount * part
            
            s, a, r, term, s1 = player.makeTurn(epsilon)
            expBuffer.add(s, a, r, term, s1)
            sampleCountTotal += 1
            episodeLengths[-1] += 1

            if term:
                episodeCountTotal += 1
                episodeLengths.append(0)
                episodeLengths = episodeLengths[-50:]

                if sampleCountTotal - sampleLastReport > reportInterval:
                    timeCur = time.perf_counter()
                    speed = (sampleCountTotal - sampleLastReport) / (timeCur - timeLastReport)
                    timeLastReport = timeCur
                    sampleLastReport += reportInterval
                    epLengthAvg = np.mean(episodeLengths)

                    evalScMin, evalScAvg, evalScMax, evalToMin, evalToAvg, evalToMax = playSomeGames(game, agentNet, EVAL_GAMES)

                    if sampleCountTotal >= initial_exp_gathering:
                        writer.add_scalar('Training/Eps', epsilon, sampleCountTotal)
                        writer.add_scalar('Training/Speed', speed, sampleCountTotal)
                        writer.add_scalar('Training/Episode Length', epLengthAvg, sampleCountTotal)
                        writer.add_scalar('Training/Eval Score', evalScAvg, sampleCountTotal)
                        writer.add_scalar('Training/Eval Score Min', evalScMin, sampleCountTotal)
                        writer.add_scalar('Training/Eval Score Max', evalScMax, sampleCountTotal)
                        writer.add_scalar('Training/Eval Total Score', evalToAvg, sampleCountTotal)
                        writer.add_scalar('Training/Eval Total Score Min', evalToMin, sampleCountTotal)
                        writer.add_scalar('Training/Eval Total Score Max', evalToMax, sampleCountTotal)
                        if lossCnt > 0:
                            writer.add_scalar('Training/Loss', lossAcc/lossCnt, sampleCountTotal)

                    print(f'Played {sampleLastReport} steps ({episodeCountTotal} episodes) ({speed:8.2f} samples/s): Average steps {epLengthAvg:7.2f}, Evaluation score {evalScMin:2}, {evalScAvg:4.1f}, {evalScMax:2}, total {evalToMin:5}, {evalToAvg:7.1f}, {evalToMax:5}')
                    lossAcc = lossCnt = 0
                
                game.reset()

            if sampleCountTotal < initial_exp_gathering:
                continue

            if sampleCountTotal - lastSyncSample > targetSyncInterval:
                lastSyncSample += targetSyncInterval
                targetNet.load_state_dict(agentNet.state_dict())

            optim.zero_grad()

            states, actions, rewards, terms, newStates = expBuffer.sample(batch_size)
            states_t = agentNet.prepareInputs(states) #torch.Tensor(agentNet.prepareInput(s) for s in states).to(device)
            newStates_t = agentNet.prepareInputs(newStates) #torch.Tensor([agentNet.prepareInput(s) for s in newStates]).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            rewards_t = torch.from_numpy(rewards).to(device)
            terms_t = torch.from_numpy(terms).to(device)

            stateActionQs = agentNet(states_t)
            stateActionQs = torch.gather(stateActionQs, 1, actions_t.unsqueeze(-1)).squeeze(-1)

            nextStateQs = targetNet(newStates_t).max(1)[0]
            nextStateQs[terms_t] = 0.0
            nextStateQs = nextStateQs.detach()

            rewards_t = nextStateQs * gamma + rewards_t
            loss = lossFunc(stateActionQs, rewards_t)

            loss.backward()
            optim.step()

            lossAcc += loss.item()
            lossCnt += 1

    except KeyboardInterrupt:
        print(f'Playing stopped after {sampleCountTotal} steps ({episodeCountTotal} episodes).')
        torch.save(agentNet.state_dict(), os.path.join('models', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+f' SC {sampleCountTotal}'))
    
    writer.close()


def playThroughStdin(game: Game2048):
    # 0 - left, 1 - up, 2 - right, 3 - down
    directions = { 'W': 1, 'S': 3, 'A': 0, 'D': 2}

    game.reset()
    moveCount = 0

    try:
        while True:
            print(game.boardAsString())
            inStr = input('Enter direction of swipe: ')
            dir = directions[inStr.upper()]
            score, ended, valid = game.swipe(dir)

            if not valid:
                print('Invalid move, try again...')
                continue

            moveCount += 1

            if ended:
                print(f'Game Over. Score: {2**game.score} after {moveCount} turns.')
                return
            else:
                print(f'Swipe #{moveCount} gave {score} points.')
        
    except KeyboardInterrupt:
        print('Game aborted.')


if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()
    # playSomeRandomGames(game)
    # playThroughStdin(game)
    # exit()

    agentNet = AgentNet()
    #agentNet.load_state_dict(torch.load('models/2021-04-06_23-13-56 SC 3281576'))

    targetNet = AgentNet()
    targetNet.load_state_dict(agentNet.state_dict())

    player = AgentPlayer(agentNet, game)

    #playAndPrintEpisode(player, 0)

    playAndLearn(agentNet, targetNet, player)
    
    print('Done!')
