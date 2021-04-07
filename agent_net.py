import torch
import torch.nn as nn
import numpy as np

class AgentNet(torch.nn.Module):
    device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self):
        super(AgentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4*4*12, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,4)
        ).to(self.device)
    
    # input: batch of 4x4 boards with 12 layers, output: estimated Q values for each direction of swipe
    def forward(self, x):
        x = self.net(x.view(x.size()[0], -1))
        return x

    # convert a batch of 4x4 boards to NN input batch
    def prepareInputs(self, boards: np.ndarray):
        if len(boards.shape) == 2:
            boards = np.expand_dims(boards, 0)
        
        dims = boards.shape
        inputs = torch.zeros((dims[0],4,4,12), dtype=torch.float32)
        for b in range(dims[0]):
            for i in range(0, 4):
                for j in range(0, 4):
                    inputs[b,i,j,boards[b][i][j]] = 1.0
        
        return inputs.to(self.device)
