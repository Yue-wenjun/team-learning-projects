import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from LSTM.LSTM_model import LSTMModel

def generate_temp_data():
    days = np.arange(1,366).reshape(-1,1) #生成特征：第一天到第三百六十五天
    temp = 5 + 30*np.sin(days*2*np.pi/365)+np.random.randn(365,1)*2 #生成气温：基础正弦波动（季节变化）+ 随机噪声（日常变化）
    data = np.hstack((days, temp)) #合并为（365,2）的数组

    max_vals = data.max(axis=0)
    min_vals = data.min(axis=0)
    data_scaled = (data-min_vals)/(max_vals-min_vals+1e-8) #所有特征值缩放到0,1
    return data_scaled,data,days

seq_len = 7 #序列长度

epochs = 150
model = LSTMModel()
criterion = nn.MSELoss()  # 回归任务使用均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
train_losses = []
for epoch in range(epochs):

    data_scaled,_,__ = generate_temp_data()

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i - seq_len:i])  # 输入前七天
        y.append(data_scaled[i, 1])  # 输出第八天
    X = torch.tensor(X, dtype=torch.float32)  # 形状（358,7,2）
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 形状（358,1）

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    total_loss = 0
    for batch_X, batch_y in dataloader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*batch_X.size(0) #平均损失乘批次数量
        avg_loss = total_loss/len(dataset)
        train_losses.append(avg_loss)
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch}, loss: {avg_loss:.3f}')

model.eval()
_,data,days = generate_temp_data()
temp_min = data[:,1].min()
temp_max = data[:,1].max()
with torch.no_grad():
    pred_scaled = model(X)
    pred_temp = np.zeros((358,2))
    pred_temp[:,1]=pred_scaled.squeeze()
    pred_temp = pred_temp*(temp_max-temp_min)+temp_min

for i in range(seq_len,len(days-seq_len)):
    print(f"{days[i][0]}\t{data[i][1]}\t{pred_temp[i-seq_len][1]}")
