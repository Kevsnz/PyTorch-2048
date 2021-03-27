from game import Game2048
import random

if __name__ == '__main__':
    print('Welcome!')

    dirs = ['left','up','right','down']

    game = Game2048()

    for i in range(20):
        print(f'{i}: Placing new number...')
        game.placeNewNumber()
        print(game.boardAsString())

        dir = random.randint(0,3)
        print(f'{i}: Swiping {dirs[dir]}...')
        game.swipe(dir)
        print(game.boardAsString())
    
    print('Done!')
