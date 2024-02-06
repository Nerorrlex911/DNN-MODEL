# DNN-Model

使用Morgan fingerprint 加载SMILES数据 分割训练集70%、验证集20%、测试集10%

并使用一个基于PyTorch的深度神经网络（结构为512, 256, 128三层全连接层，在输入层和第一层、第一层和第二层之间dropout处理(比例为0.25)）