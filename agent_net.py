import torch
import torch.nn as nn
import torch.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AgentNet(torch.nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        self.L1 = nn.Linear(4*4*12, 128).to(device)
        self.L2 = nn.Linear(128,4).to(device)
    
    # input: 4x4 board with 12 layers, output: direction of swipe
    def forward(self, x):
        x = F.relu(self.L1(torch.flatten(x)))
        x = torch.sigmoid(self.L2(x))
        return x
    
    # convert 4x4 board to NN input
    def prepareInput(self, board):
        input = torch.zeros((4,4,12), dtype=torch.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                input[i,j,board[i][j]] = 1.0
        
        return input.to(device)
