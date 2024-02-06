from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
class MolDataSet(Dataset):
    def __init__(self,data_path) -> None:
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass


def finger(mol):
    """
    计算分子的Morgan指纹
    返回一个1024位的01数组
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)   #计算1024位的指纹
    fp = fp.ToBitString()
    fp = np.array(list(fp)).astype('int8')
    return fp