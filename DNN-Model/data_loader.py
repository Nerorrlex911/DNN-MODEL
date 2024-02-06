from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
class MolDataSet(Dataset):
    def __init__(self,data_path) -> None:
        assert os.path.exists(data_path), f"{data_path} does not exist"
        smiles = pd.read_csv(data_path,names=['smiles','mol_id','activity'])  #读取smiles文件
        smiles = smiles.sample(frac=1)  #打乱数据
        mols = [Chem.MolFromSmiles(smi) for smi in smiles['smiles']]
        self.data = torch.from_numpy(np.array([finger(m) for m in mols]))
        self.label = torch.from_numpy(np.array(smiles['activity']))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def finger(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)   #计算1024位的指纹
    fp = fp.ToBitString()
    fp = np.array(list(fp)).astype('int8')
    return fp