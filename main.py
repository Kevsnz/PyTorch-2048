from game import Game2048
from agent_net import AgentNet
from agent_player import AgentPlayer
import random

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


if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()

    for i in range(150):
        game.reset()
        turns, score, won = silentPlayout(game)

        if won:
            print(f'Victory! Target score of {2**score} reached after {turns} turns.')
        else:
            print(f'Game Over! No space for new number after {turns} turns (final score: {2**score}).')

    # agentNet = AgentNet()
    # targetNet = AgentNet()
    # player = AgentPlayer(agentNet, game)

    # s,a,r,e = player.playEpisode()

    # for i in range(0,len(s)):
    #     print(f'State: {s[i]}')
    #     print(f'Action: {a[i]}')
    #     print(f'Reward: {r[i]}')
    #     print(f'Ended: {e[i]}')
    
    
    print('Done!')
