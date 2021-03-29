from game import Game2048
from agent_net import AgentNet
from agent_player import AgentPlayer
from experience_buffer import ExperienceBuffer
import random
import time
import torch

exp_capacity = 100000
epsilon_initial = 1.0
epsilon_final = 0.02
epsilon_decay_time = 500000
epsilon_decay_amount = epsilon_initial - epsilon_final
initial_exp_gathering = 5000
batch_size = 16
gamma = 0.95

def silentPlayout(game):
    turn = 0

    while True:
        turn += 1
        dir = random.randint(0,3)
        _, ended = game.swipe(dir)
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


def playAndPrintEpisode(player):
    s, a, r, e, _ = player.playEpisode()

    for i in range(0,len(s)):
        print(f'State: {s[i]}')
        print(f'Action: {a[i]}')
        print(f'Reward: {r[i]}')
        print(f'Ended: {e[i]}')


def playAndLearn(agentNet, targetNet, player):
    expBuffer = ExperienceBuffer(exp_capacity)
    device = AgentNet.device

    reportInterval = 5000

    sampleCount = 0
    sampleLast = 0
    episodeCount = 0
    timeLast = time.perf_counter()
    epsilon = epsilon_initial

    lossFunc = torch.nn.MSELoss()
    optim = torch.optim.Adam(agentNet.parameters(), 0.001)

    stepCount = 0
    epCount = 0

    try:
        while True:
            if sampleCount > initial_exp_gathering:
                part = min(1.0, (sampleCount - initial_exp_gathering) / epsilon_decay_time)
                epsilon = epsilon_initial - epsilon_decay_amount * part
            
            s, a, r, e, s1 = player.playEpisode(epsilon)
            expBuffer.add(s, a, r, e, s1)
            sampleCount += len(s)
            episodeCount += 1

            stepCount += len(s)
            epCount += 1

            if sampleCount - sampleLast > reportInterval:
                sampleLast += reportInterval
                timeCur = time.perf_counter()
                speed = reportInterval / (timeCur - timeLast)
                timeLast = timeCur
                print(f'Played {sampleLast} steps ({episodeCount} episodes) ({speed} samples/s): Average step count {stepCount/epCount}')
                stepCount = 0
                epCount = 0

            if sampleCount < initial_exp_gathering:
                continue

            states, actions, rewards, terms, newStates = expBuffer.sample(batch_size)
            states_t = agentNet.prepareInputs(states) #torch.Tensor(agentNet.prepareInput(s) for s in states).to(device)
            newStates_t = agentNet.prepareInputs(newStates) #torch.Tensor([agentNet.prepareInput(s) for s in newStates]).to(device)
            actions_t = torch.tensor(actions, dtype=torch.int64).to(device)
            rewards_t = torch.tensor(rewards).to(device)
            terms_t = torch.tensor(terms, dtype=torch.bool).to(device)

            stateActionQs = agentNet(states_t)
            stateActionQs = torch.gather(stateActionQs, 1, actions_t.unsqueeze(-1)).squeeze(-1)

            q1_t = targetNet(newStates_t).max(1)[0]
            q1_t[terms_t] = 0.0
            rewards_t = q1_t.detach() * gamma + rewards_t
            loss = lossFunc(stateActionQs, rewards_t)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if episodeCount % 5 == 4:
                targetNet.load_state_dict(agentNet.state_dict())


    except KeyboardInterrupt:
        print(f'Playing stopped after {sampleCount} steps ({episodeCount} episodes).')


if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()
    # playSomeRandomGames(game)

    agentNet = AgentNet()
    targetNet = AgentNet()
    player = AgentPlayer(agentNet, game)

    # playAndPrintEpisode(player)

    playAndLearn(agentNet, targetNet, player)
    
    print('Done!')
