import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import Game2048

class AgentNet(torch.nn.Module):
    device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self):
        super(AgentNet, self).__init__()
        self.netPre = nn.Sequential(
            nn.Linear(4*4*12, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU()
        ).to(self.device)

        self.netVal = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        ).to(self.device)
        
        self.netAdv = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,4)
        ).to(self.device)
    
    # input: batch of 4x4 boards with 12 layers, output: estimated Q values for each direction of swipe
    def forward(self, x):
        x = self.netPre(x.view(x.size()[0], -1))
        val = self.netVal(x)
        adv = self.netAdv(x)
        return val + adv - adv.mean()

    # convert a batch of 4x4 boards to NN input batch
    def prepareInputs(self, boards: np.ndarray):
        if len(boards.shape) == 2:
            boards = np.expand_dims(boards, 0)
        
        inputs = F.one_hot(torch.as_tensor(boards, dtype=torch.int64), Game2048.TARGET_SCORE+1)
        return inputs.type(torch.FloatTensor).to(self.device)
