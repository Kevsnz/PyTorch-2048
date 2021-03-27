import random
import numpy as np


class Game2048:
    target_score = 11 # 2 ** 11 = 2048

    def __init__(self):
        self.numbers = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.reset()
    

    def reset(self):
        self.board = np.zeros(16).reshape(4, 4) # [row][col]
        self.score = 0
    

    def placeNewNumber(self):
        i, j = self.getFreePlace()

        if i == None or j == None:
            raise Exception('No free space left!')

        self.board[i][j] = 1
    

    def getFreePlace(self):
        freePlaces = []

        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    freePlaces.append([i, j])
        
        if len(freePlaces) == 0:
            return
        
        idx = random.randint(0, len(freePlaces)-1)
        return freePlaces[idx][0], freePlaces[idx][1]
    

    def swipe(self, dir): # 0 - left, 1 - up, 2 - right, 3 - down
        # Flip the board around so the direction needed for the swipe points to the right
        if dir == 0:
            self.board = np.fliplr(self.board)
        elif dir == 1:
            self.board = np.fliplr(self.board.transpose())
        elif dir == 3:
            self.board = self.board.transpose()

        # Do the swipe to the right
        for i in range(0, 4): # each line
            for j in range(2, -1, -1): # from right to left
                self.sweepRight(i, j) # sweep to the right

        # Unflip board back to original orientation
        if dir == 0:
            self.board = np.fliplr(self.board)
        elif dir == 1:
            self.board = np.fliplr(self.board).transpose()
        elif dir == 3:
            self.board = self.board.transpose()


    def sweepRight(self, i, j):
        if j == 3:
            return

        if self.board[i][j] == 0:
            return

        if self.board[i][j+1] == 0:
            self.board[i][j+1] = self.board[i][j]
            self.board[i][j] = 0
            self.sweepRight(i, j+1)
        elif self.board[i][j+1] == self.board[i][j]:
            self.board[i][j+1] = self.board[i][j] + 1
            self.board[i][j] = 0

            if self.score < self.board[i][j+1]:
                self.score = self.board[i][j+1]
            
            self.sweepRight(i, j+1)


    def boardAsString(self):
        strBoard = ''

        for i in range(4):
            for j in range(4):
                strBoard = strBoard + str(self.numbers[int(self.board[i][j])]).rjust(6)
            
            strBoard = strBoard + '\n'
        
        return strBoard[0:-1]
