from game import Game2048
import random

if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()

    for i in range(1024):
        print(f'{i+1}: Placing new number...')

        try:
            game.placeNewNumber()
        except Exception:
            print('No space for new number. Game Over!')
            break

        print(game.boardAsString())

        dir = random.randint(0,3)
        print(f'{i+1}: Swiping {dirs[dir]}...')
        game.swipe(dir)
        print(game.boardAsString())
        if game.score == game.target_score:
            print('Target score reached. Victory!')
    
    print('Done!')
