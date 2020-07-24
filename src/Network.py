import collections
import os
import sys
import numpy as np
import torch
from torch import reshape
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()

        self.output_size = output_size

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Dropout2d(p=0.20)
            )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout2d(0.25)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
        )

        self.fc = nn.Sequential(
            #全結合層
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, self.output_size)
        )
 

    def forward(self, x):  

        x = x.to("cuda")
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)

        out = out.view(out.size()[0], -1)
        
        out = self.fc(out)
        out = F.softmax(out)

        return out
