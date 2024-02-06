from torch import nn
import torch
import os
import pandas as pd
from data_loader import MolDataSet
from torch.utils.data import DataLoader

class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.dropout1 = nn.Dropout(0.25)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.dropout2 = nn.Dropout(0.25)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
