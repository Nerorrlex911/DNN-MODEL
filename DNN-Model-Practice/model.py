from torch import nn
import torch
import os
import pandas as pd
from data_loader import MolDataSet
from torch.utils.data import DataLoader

class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        """
        初始化模型的结构
        :param in_dim: 输入维度
        :param n_hidden_1: 第一个隐藏层维度
        :param n_hidden_2: 第二个隐藏层维度
        :param out_dim: 输出维度
        """
        pass

    def forward(self, x):
        return x
    
