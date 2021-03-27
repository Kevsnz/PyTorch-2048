
class Game2048:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.score = 0

    def placeNewNumber(self):
        pass

    def canPlaceNewNumber(self):
        return False
    
    def swipe(self, dir):
        pass

    def print(self):
        strBoard = ''

        for i in range(4):
            for j in range(4):
                strBoard = strBoard + str(self.board[i][j]).rjust(6)
            
            strBoard = strBoard + '\n'
        
        return strBoard[0:-1]

