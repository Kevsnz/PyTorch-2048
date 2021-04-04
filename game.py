import random
import numpy as np


class Game2048:
    TARGET_SCORE = 11 # 2 ** 11 = 2048
    FOUR_PROBABILITY = 0.1

    def __init__(self):
        self.numbers = [0]
        for i in range(self.TARGET_SCORE):
            self.numbers.append(2**(i+1))
        self.reset()
    

    def reset(self):
        self.board = np.zeros(16, dtype=np.int16).reshape(4, 4) # [row][col]
        self.score = 0
        self.totalScore = 0
        self.placeNewNumber()
        self.placeNewNumber()
    
    
    def clearMergeBoard(self):
        self.mergeBoard = np.zeros(16, dtype=np.bool8).reshape(4, 4)
    

    # Returns whether game is over (no space for new number)
    def placeNewNumber(self):
        i, j = self.getFreePlace()

        if i == None or j == None:
            return True

        self.board[i][j] = 2 if random.random() < self.FOUR_PROBABILITY else 1
        return not self.isValidSwipesAvailable()
    

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
    

    def isValidSwipesAvailable(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return True

                if i < 3 and self.board[i][j] == self.board[i+1][j]:
                    return True
                
                if j < 3 and self.board[i][j] == self.board[i][j+1]:
                    return True
        
        return False

    
    # dir: 0 - left, 1 - up, 2 - right, 3 - down
    # Returns swipe score, whether the game is ended and whether swipe is valid
    def swipe(self, dir):
        # Flip the board around so the direction needed for the swipe points to the right
        if dir == 0:
            self.board = np.fliplr(self.board)
        elif dir == 1:
            self.board = np.fliplr(self.board.transpose())
        elif dir == 3:
            self.board = self.board.transpose()

        turnScore = 0
        isValidMove = False
        self.clearMergeBoard()
        # Do the swipe to the right
        for i in range(0, 4): # each line
            for j in range(2, -1, -1): # from right to left
                score, valid = self.sweepRight(i, j) # sweep to the right
                turnScore += score
                isValidMove = isValidMove or valid

        self.totalScore += turnScore

        # Unflip board back to original orientation
        if dir == 0:
            self.board = np.fliplr(self.board)
        elif dir == 1:
            self.board = np.fliplr(self.board).transpose()
        elif dir == 3:
            self.board = self.board.transpose()
        
        if not isValidMove:
            return 0, False, False

        if self.score == self.TARGET_SCORE:
            return turnScore, True, True
        
        gameOver = self.placeNewNumber()
        if gameOver:
            return -11, True, True
        return turnScore, False, True


    # Returns score addition as a result of a tile move and if there was a board change
    def sweepRight(self, i, j):
        if j == 3:
            return 0, False

        if self.board[i][j] == 0:
            return 0, False

        if self.board[i][j+1] == 0:
            self.board[i][j+1] = self.board[i][j]
            self.board[i][j] = 0
            score, _ = self.sweepRight(i, j+1)
            return score, True

        if self.board[i][j+1] == self.board[i][j] and not self.mergeBoard[i][j+1]:
            self.board[i][j+1] = self.board[i][j] + 1
            self.board[i][j] = 0
            self.mergeBoard[i][j+1] = True

            if self.score < self.board[i][j+1]:
                self.score = self.board[i][j+1]
            
            score = self.board[i][j+1]
            #score2, _ = self.sweepRight(i, j+1)
            #return score + score2, True
            return score, True
        
        return 0, False


    def boardAsString(self):
        strBoard = ''

        for i in range(4):
            for j in range(4):
                strBoard = strBoard + str(self.numbers[int(self.board[i][j])]).rjust(6)
            
            strBoard = strBoard + '\n'
        
        return strBoard[0:-1]

