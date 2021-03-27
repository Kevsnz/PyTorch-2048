from game import Game2048
import random

def silentPlayout(game):
    turn = 0

    while True:
        try:
            game.placeNewNumber()
        except Exception:
            return turn, False
        
        turn += 1
        dir = random.randint(0,3)
        game.swipe(dir)
        if game.score == game.target_score:
            return turn, True


if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()

    for i in range(150):
        game.reset()
        turns, won = silentPlayout(game)

        if won:
            print(f'Victory! Target score reached after {turns} turns.')
        else:
            print(f'Game Over! No space for new number after {turns} turns.')
    
    print('Done!')
