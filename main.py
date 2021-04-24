import random
import time
import torch
import torch.nn.utils
import datetime
import os
import numpy as np
import collections
from game import Game2048
from agent_net import AgentNet
from agent_player import AgentPlayer
from experience_unroller import ExperienceUnroller
from tensorboardX import SummaryWriter

evaluation_interval = 2000

BATCH_SIZE = 128
GAMMA = 0.99
learning_rate = 0.001
GRAD_CLIP = 0.1
EXP_UNROLL_STEPS = 4

EVAL_GAMES = 10
ENT_BETA = 0.01

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
    s, a, r, e, _ = player.playEpisode()
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
        agent.playEpisode(True)
        avgScore += game.score
        maxScore = max(maxScore, game.score)
        minScore = min(minScore, game.score)
        avgTotal += game.totalScore
        maxTotal = max(maxTotal, game.totalScore)
        minTotal = min(minTotal, game.totalScore)
    
    return minScore, avgScore / count, maxScore, minTotal, avgTotal / count, maxTotal


def playAndLearn(agentNet, player):
    expUnroller = ExperienceUnroller(EXP_UNROLL_STEPS, GAMMA)
    qGamma = GAMMA ** (EXP_UNROLL_STEPS + 1)
    device = AgentNet.device

    writer = SummaryWriter(logdir=os.path.join('tensorboard', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    reportInterval = 2000

    sampleCountTotal = 0
    sampleLastReport = 0
    episodeCountTotal = 0
    timeLastReport = time.perf_counter()

    mseFunc = torch.nn.MSELoss().to(device)
    logSoftMaxFunc = torch.nn.LogSoftmax(dim=1).to(device)
    softMaxFunc = torch.nn.Softmax(dim=1).to(device)
    optim = torch.optim.Adam(agentNet.parameters(), lr=learning_rate, eps=1e-3)

    lossAcc = 0
    lossPolicyAcc = 0
    lossValueAcc = 0
    lossEntropyAcc = 0
    gradL2Acc = 0
    gradMaxAcc = 0
    gradVarAcc = 0
    lossCnt = 0
    epLen = 0
    episodeLengths = collections.deque(maxlen=20)
    evalScMin = evalScAvg = evalScMax = evalToMin = evalToAvg = evalToMax = 0

    states = []
    actions = []
    rewards = []
    nonTermIdxs = []
    nextStates = []

    try:
        while True:
            s, a, r, term, s1 = player.makeTurn()
            sampleCountTotal += 1
            epLen += 1

            if term:
                episodeCountTotal += 1
                episodeLengths.append(epLen)
                epLen = 0
                game.reset()
            
            s, a, r, term, s1 = expUnroller.add(s, a, r, term, s1)
            if s is not None:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                if not term:
                    nonTermIdxs.append(len(states) - 1)
                    nextStates.append(s1)
            
            if sampleCountTotal - sampleLastReport > reportInterval:
                timeCur = time.perf_counter()
                speed = (sampleCountTotal - sampleLastReport) / (timeCur - timeLastReport)
                timeLastReport = timeCur
                sampleLastReport += reportInterval
                epLengthAvg = np.mean(episodeLengths)

                writer.add_scalar('Training/Speed', speed, sampleCountTotal)
                writer.add_scalar('Training/Episode Length', epLengthAvg, sampleCountTotal)
                if lossCnt > 0:
                    writer.add_scalar('Training/Loss', lossAcc/lossCnt, sampleCountTotal)
                    writer.add_scalar('Training/Loss Value', lossValueAcc/lossCnt, sampleCountTotal)
                    writer.add_scalar('Training/Loss Policy', lossPolicyAcc/lossCnt, sampleCountTotal)
                    writer.add_scalar('Training/Loss Entropy', lossEntropyAcc/lossCnt, sampleCountTotal)
                    writer.add_scalar('Training/Grad L2', gradL2Acc/lossCnt, sampleCountTotal)
                    writer.add_scalar('Training/Grad Max', gradMaxAcc/lossCnt, sampleCountTotal)
                    writer.add_scalar('Training/Grad Var', gradVarAcc/lossCnt, sampleCountTotal)

                print(f'Played {sampleLastReport} steps ({episodeCountTotal} episodes) ({speed:8.2f} samples/s): Average steps {epLengthAvg:7.2f}, Evaluation score {evalScMin:2}, {evalScAvg:4.1f}, {evalScMax:2}, total {evalToMin:5}, {evalToAvg:7.1f}, {evalToMax:5}')
                lossAcc = lossValueAcc = lossPolicyAcc = lossEntropyAcc = gradL2Acc = gradMaxAcc = gradVarAcc = lossCnt = 0
            
            if sampleCountTotal % evaluation_interval == 0:
                evalScMin, evalScAvg, evalScMax, evalToMin, evalToAvg, evalToMax = playSomeGames(Game2048(), agentNet, EVAL_GAMES)

                writer.add_scalar('Evaluation/Eval Score Avg', evalScAvg, sampleCountTotal)
                writer.add_scalar('Evaluation/Eval Score Min', evalScMin, sampleCountTotal)
                writer.add_scalar('Evaluation/Eval Score Max', evalScMax, sampleCountTotal)
                writer.add_scalar('Evaluation/Eval Total Score Avg', evalToAvg, sampleCountTotal)
                writer.add_scalar('Evaluation/Eval Total Score Min', evalToMin, sampleCountTotal)
                writer.add_scalar('Evaluation/Eval Total Score Max', evalToMax, sampleCountTotal)

            if len(states) < BATCH_SIZE:
                continue
            
            states_t = agentNet.prepareInputs(np.array(states, copy=False))
            actions_t = torch.LongTensor(actions).to(device)

            rewards_np = np.array(rewards, dtype=np.float32)
            if nonTermIdxs:
                nextStates_t = agentNet.prepareInputs(np.array(nextStates, copy=False))
                nextStateVals_t = agentNet(nextStates_t)[1]
                rewards_np[nonTermIdxs] += qGamma * nextStateVals_t.data.cpu().numpy()[:, 0]
            
            refVals_t = torch.from_numpy(rewards_np).to(device)
            
            optim.zero_grad()
            logits_t, value_t = agentNet(states_t)

            lossValue_t = mseFunc(value_t.squeeze(-1), refVals_t)

            logProb_t = logSoftMaxFunc(logits_t)
            adv_t = refVals_t - value_t.squeeze(-1)
            logProbActions_t = adv_t.detach() * logProb_t[range(BATCH_SIZE), actions_t.squeeze(-1)]
            lossPolicy_t = -logProbActions_t.mean()

            prob_t = softMaxFunc(logits_t)
            lossEntropy_t = -ENT_BETA * (prob_t * logProb_t).sum(dim=1).mean()

            lossPolicy_t.backward(retain_graph=True)
            grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                for p in agentNet.parameters()
                if p.grad is not None])
            loss_t = lossValue_t * 0.5 - lossEntropy_t
            loss_t.backward()

            torch.nn.utils.clip_grad_norm_(agentNet.parameters(), GRAD_CLIP)
            optim.step()

            lossAcc += loss_t.item() + lossPolicy_t.item()
            lossValueAcc += lossValue_t.item()
            lossPolicyAcc += lossPolicy_t.item()
            lossEntropyAcc += lossEntropy_t.item() / ENT_BETA
            gradL2Acc += np.sqrt(np.mean(np.square(grads)))
            gradMaxAcc += np.max(np.abs(grads))
            gradVarAcc += np.var(grads)
            lossCnt += 1

            states.clear()
            actions.clear()
            rewards.clear()
            nonTermIdxs.clear()
            nextStates.clear()
            expUnroller.clear()

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


def loadModel(net: AgentNet, filename: str):
    net.load_state_dict(torch.load(('models/' + filename)))
    print(f'Loaded parameters from file {filename}')


if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()
    # playSomeRandomGames(game)
    # playThroughStdin(game)
    # exit()

    agentNet = AgentNet()
    #loadModel(agentNet, '2021-04-18_17-43-54 SC 2674000')

    player = AgentPlayer(agentNet, game)

    #playAndPrintEpisode(player, 0)

    playAndLearn(agentNet, player)
    
    print('Done!')
