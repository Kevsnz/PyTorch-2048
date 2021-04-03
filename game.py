import random
import numpy as np


class Game2048:
    target_score = 11 # 2 ** 11 = 2048

    def __init__(self):
        self.numbers = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.reset()
    

    def reset(self):
        self.board = np.zeros(16, dtype=np.int16).reshape(4, 4) # [row][col]
        self.score = 0
        self.placeNewNumber()
    

    # Returns whether game is over (no space for new number)
    def placeNewNumber(self):
        i, j = self.getFreePlace()

        if i == None or j == None:
            return True

        self.board[i][j] = 1
        return False
    

    def getFreePlace(self):
        freePlaces = []

        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    freePlaces.append([i, j])
        
        if len(freePlaces) == 0:
            return None, None
        
        idx = random.randint(0, len(freePlaces)-1)
        return freePlaces[idx][0], freePlaces[idx][1]
    

    # 0 - left, 1 - up, 2 - right, 3 - down
    def swipe(self, dir):
        # Flip the board around so the direction needed for the swipe points to the right
        if dir == 0:
            self.board = np.fliplr(self.board)
        elif dir == 1:
            self.board = np.fliplr(self.board.transpose())
        elif dir == 3:
            self.board = self.board.transpose()

        turnScore = 0
        # Do the swipe to the right
        for i in range(0, 4): # each line
            for j in range(2, -1, -1): # from right to left
                turnScore += self.sweepRight(i, j) # sweep to the right

        # Unflip board back to original orientation
        if dir == 0:
            self.board = np.fliplr(self.board)
        elif dir == 1:
            self.board = np.fliplr(self.board).transpose()
        elif dir == 3:
            self.board = self.board.transpose()
        
        if self.score == self.target_score:
            return turnScore, True
        
        gameOver = self.placeNewNumber()
        if gameOver:
            return -11, True
        return turnScore, False


    def sweepRight(self, i, j):
        if j == 3:
            return 0

        if self.board[i][j] == 0:
            return 0

        if self.board[i][j+1] == 0:
            self.board[i][j+1] = self.board[i][j]
            self.board[i][j] = 0
            return self.sweepRight(i, j+1)

        if self.board[i][j+1] == self.board[i][j]:
            self.board[i][j+1] = self.board[i][j] + 1
            self.board[i][j] = 0

            if self.score < self.board[i][j+1]:
                self.score = self.board[i][j+1]
            
            return self.board[i][j+1] + self.sweepRight(i, j+1)
        
        return 0


    def boardAsString(self):
        strBoard = ''

        for i in range(4):
            for j in range(4):
                strBoard = strBoard + str(self.numbers[int(self.board[i][j])]).rjust(6)
            
            strBoard = strBoard + '\n'
        
        return strBoard[0:-1]

