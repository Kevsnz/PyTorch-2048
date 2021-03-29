import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AgentNet(torch.nn.Module):
    device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self):
        super(AgentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4*4*12, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,4)
        ).to(self.device)
    
    # input: 4x4 board with 12 layers, output: estimated Q values for each direction of swipe
    def forward(self, x):
        x = self.net(x.view(x.size()[0], -1))
        return x
    
    # convert 4x4 board to NN input
    def prepareInput(self, board):
        input = torch.zeros((4,4,12), dtype=torch.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                input[i,j,board[i][j]] = 1.0
        
        return input.unsqueeze(0).to(self.device)

    def prepareInputs(self, boards: np.ndarray):
        dims = boards.shape
        if len(dims) == 2:
            return self.prepareInput(boards)
        
        inputs = torch.zeros((dims[0],4,4,12), dtype=torch.float32)
        for b in range(dims[0]):
            for i in range(0, 4):
                for j in range(0, 4):
                    inputs[b,i,j,boards[b][i][j]] = 1.0
        
        return inputs.to(self.device)
