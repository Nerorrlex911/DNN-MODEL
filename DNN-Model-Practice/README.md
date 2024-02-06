# DNN-Model (医药人工智能练习，附答案)

使用Morgan fingerprint 加载SMILES数据 分割训练集70%、验证集20%、测试集10%

并使用一个基于PyTorch的深度神经网络

神经网络结构：
1. 损失函数：均方误差
2. 优化算法：动量梯度下降
3. 层结构为512, 256, 128三层全连接层，在输入层和第一层、第一层和第二层之间dropout处理(比例为0.25)

要求可运行于GPU/CPU

要求兼容Windows、Linux系统

主要涉及Python模块:
1. numpy
2. [rdkit](https://rdkit.org/docs/index.html) 化学工具，用于读取SMILES、生成分子指纹
3. [pandas](https://pandas.pydata.org/docs/getting_started/index.html) 数据管理框架，用于读取csv表格数据