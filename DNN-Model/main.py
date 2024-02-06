from torch import nn
import torch
import os
import pandas as pd
from data_loader import MolDataSet
from torch.utils.data import DataLoader
from model import Neuralnetwork
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.005, help='star learning rate')   
parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(),'DNN-Model','datasets','train.smi')) 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args() 

# 定义当前模型的训练环境
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 
batch_size = opt.batch_size
lr = opt.lr
data_path = opt.data_path
epochs = opt.epochs

# 划分数据集并加载
custom_dataset = MolDataSet(data_path=data_path)
train_size = int(len(custom_dataset) * 0.7)
validate_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - validate_size - train_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validate_size, test_size])
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False )
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False ) 
print("Training set data size:", len(train_loader)*batch_size, ",Validating set data size:", len(validate_loader), ",Testing set data size:", len(test_loader)) 

if __name__ == '__main__':
    model = Neuralnetwork(1024, 512, 256, 128, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True)
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        for data in validate_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().view(-1,1))
            print(f'Validation Loss: {loss.item():.4f}')
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().view(-1,1))
            print(f'Test Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'model.ckpt')
    print("Model has been saved.")