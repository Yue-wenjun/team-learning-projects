import os
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpl
import datetime
import math
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from tifffile import astype
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

#通过爬虫获取网页信息，成都历史天气信息
# def get_url(city, start, end):
#     url_list = []
#     for year in range(start, end):
#         for month in range(1, 13):
#             y = year * 100 + month
#             url1 = "http://lishi.tianqi.com/"+city+"/"+str(y)+".html"
#             url_list.append(url1)
#     return url_list

#将请求到的数据解析为csv文件
# def get_weather_month(url,file):
#     headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SLBrowser/9.0.6.8151 SLBChan/112 SLBVPV/64-bit',
#                'Cookie':'UserId=17600891102797225; Hm_lvt_7c50c7060f1f743bccf8c150a646e90a=1760089370,1760089376,1760089652,1760089656; HMACCOUNT=E1494C523FF273B0; Hm_lpvt_7c50c7060f1f743bccf8c150a646e90a=1760089895',
#                'Connection': 'keep-alive','Host': 'lishi.tianqi.com'
#                }
#     try:
#         # 1. 发送请求并主动检测HTTP错误（如404、503）
#         response = requests.get(url, headers=headers, timeout=15)  # 延长超时到15秒，避免网络波动
#         response.raise_for_status()  # 若状态码非200（如404），直接抛出异常
#         print(f"请求{url}成功，状态码：{response.status_code}")
#         html = requests.get(url, headers=headers)
#         html.encoding = 'utf-8'
#         soup = BeautifulSoup(html.text,'lxml')
#         weather_list = soup.select('ul[class="thrui"]')
#         for weather in weather_list:
#             ul_list = weather.select('li')
#             if not ul_list:
#                 print(f"{url} 未找到日期数据（li标签）")
#                 continue
#             for li in ul_list:
#                 div_list = li.select('div')
#                 try:
#                     dates = div_list[0].string[:10]
#                 except:
#                     dates = 'error'
#                 try:
#                     maxTem = div_list[1].string
#                 except:
#                     maxTem = 'error'
#                 try:
#                     minTem = div_list[2].string
#                 except:
#                     minTem = 'error'
#                 try:
#                     weathers = div_list[3].string
#                 except:
#                     weathers = 'error'
#                 try:
#                     wind = div_list[4].string
#                 except:
#                     wind = 'error'
#                 if(weathers==None):weathers = '阵雨'
#                 f = open(file, 'a')
#                 I = dates+','+maxTem+','+minTem+','+weathers+','+wind+'\n'
#                 f.write(I)
#                 f.close()
#     except Exception as e:
#         print(f"{url} 抓取失败：{type(e).__name__} - {str(e)}")

#将原始时间序列转换为输入-目标对的形式，划分训练集与验证集
def Create_Sequences_One(data,sequence_length,slip_size):
    data_len = len(data)
    data_slip = int(slip_size * data_len)

    x = torch.zeros(data_len-sequence_length, sequence_length,1)
    y = torch.zeros(data_len-sequence_length, 1)

    for i in range(data_len-sequence_length-2):
        x[i] = data[i:i+sequence_length].reshape(-1,1)
        y[i] = data[i+sequence_length]

    train_X = x[:data_slip]
    train_y = y[:data_slip]
    valid_X = x[data_slip:]
    valid_y = y[data_slip:]

    return train_X, train_y, valid_X, valid_y

#创建能够参与训练的数据集
class CreateDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]
        return X, Y

#加载器，按批次加载数据
def DataLoaders(batch_size,train_x,train_y,valid_x,valid_y):
    train_Set = CreateDataset(train_x,train_y)
    valid_Set = CreateDataset(valid_x,valid_y)
    train_loader = DataLoader(train_Set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_Set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

#模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,output_dim):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:,-1,:])
        return out

#按轮次训练与验证，计算损失，存储结果，更新参数
def train_model(model,train_loader,valid_loader,loss_fn,optimizer,num_epochs,device):
    train_losses = []
    valid_losses = []
    train_mse = []
    valid_mse = []
    train_rmse = []
    valid_rmse = []
    train_mae = []
    valid_mae = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        for input,targets in train_loader:
            input, targets = input.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_predictions.extend(outputs.cpu().detach().numpy())
            train_targets.extend(targets.cpu().detach().numpy())
        #train_predictions = scaler.inverse_transform(np.array(train_prediction).reshape(-1,1))
        #train_targets = scaler.inverse_transform(np.array(train_targets).reshape(-1,1))
        model.eval()
        valid_loss = 0.0
        valid_predictions = []
        valid_targets = []
        with torch.no_grad():
            for input,targets in valid_loader:
                input, targets = input.to(device), targets.to(device)
                outputs = model(input)
                loss = loss_fn(outputs, targets)
                valid_loss += loss.item()
                valid_predictions.extend(outputs.cpu().detach().numpy())
                valid_targets.extend(targets.cpu().detach().numpy())
        #valid_predictions = scaler.inverse_transform(np.array(valid_prediction).reshape(-1,1))
        #vaild_targets = scaler.inverse_transform(np.array(valid_targets).reshape(-1,1))
        train_loss /=len(train_loader)
        valid_loss /=len(valid_loader)
        train_pred_np = np.array(train_predictions).reshape(-1, 1)
        train_tar_np = np.array(train_targets).reshape(-1, 1)
        train_mse_value = F.mse_loss(torch.tensor(train_pred_np), torch.tensor(train_tar_np))
        valid_pred_np = np.array(valid_predictions).reshape(-1, 1)
        valid_tar_np = np.array(valid_targets).reshape(-1, 1)
        valid_mse_value = F.mse_loss(torch.tensor(valid_pred_np), torch.tensor(valid_tar_np))
        # train_mse_value = F.mse_loss(torch.tensor(train_predictions), torch.tensor(train_targets))
        # valid_mse_value = F.mse_loss(torch.tensor(valid_predictions), torch.tensor(valid_targets))
        train_rmse_value = torch.sqrt(train_mse_value)
        valid_rmse_value = torch.sqrt(valid_mse_value)
        train_mae_value = F.l1_loss(torch.tensor(train_pred_np), torch.tensor(train_tar_np))
        valid_mae_value = F.l1_loss(torch.tensor(valid_pred_np), torch.tensor(valid_tar_np))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_mse.append(train_mse_value.item())
        valid_mse.append(valid_mse_value.item())
        train_rmse.append(train_rmse_value.item())
        valid_rmse.append(valid_rmse_value.item())
        train_mae.append(train_mae_value.item())
        valid_mae.append(valid_mae_value.item())

        print(f"Epoch[{epoch+1}/{num_epochs}]-Train Loss:{train_loss:.4f},Valid Loss:{valid_loss:.4f}")

    return train_losses,valid_losses,train_mse,valid_mse,train_rmse,valid_rmse,train_mae,valid_mae

#用来测试上述训练好的模型，并绘图展示预测值与真实值的关联情况，整体图和前后个100个样本的细节图
def predict_plot(model,test_loader,scaler,device):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for input,targets in test_loader:
            input = input.to(device)
            output = model(input)
            predictions.extend(output.cpu().detach().numpy())
            true_values.extend(targets.cpu().detach().numpy())
    predicted_values = scaler.inverse_transform(predictions).reshape(-1,1)
    true_values = scaler.inverse_transform(true_values).reshape(-1,1)

    plt.figure(figsize=(10,5))
    plt.plot(true_values,label='True Values',color='blue',lw=0.6)
    plt.plot(predicted_values,label='Predictions',color='red',lw=0.6)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.title('True vs Predicted Values')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(true_values[:100],label='True Values',color='blue')
    plt.plot(predicted_values[:100],label='Predictions',color='red')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.title("Predicted vs Valid(head 100)")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(true_values[len(true_values)-100:],label='True Values',color='blue')
    plt.plot(predicted_values[len(predicted_values)-100:],label='Predictions',color='red')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.title("Predicted vs Valid(tail 100)")
    plt.show()

#主程序，用爬取到的文件数据训练模型，并绘图训练中的损失指标，最后测试模型，绘图
city = "chengdu"
# time1 = 2012
# time2 = 2023
file = f"weather_{city}.csv"

# all_url = get_url(city, time1, time2)
# if not os.path.exists(file):
#     f = open(file, 'a')
#     f.write('date,maxTem,minTem,weathers,wind\n')
#     f.close()
# for url in all_url:
#     try:
#         get_weather_month(url,file)
#         time.sleep(5)
#     except:
#         print(f'{url}Error')
#     print(f'网页{url}抓取完成')
#
# df = pd.read_csv(file,encoding='gbk')
# df['maxTem'] = df['maxTem'].str.replace('℃','').astype(np.float64)
# df['minTem'] = df['minTem'].str.extract('(\d+)').astype(np.float64)

# df.index = df['date']
# df['minTem'].plot(figsize=(20,10))
# df['maxTem'].plot(figsize=(20,10))
# plt.xlabel('date',fontsize=15)
# plt.ylabel('temperature',fontsize=15)
# plt.title('Max/Min Temperature(2012-2022)',fontsize=20)
# plt.tick_params(labelsize=10)
# plt.legend()
# plt.show()

data = pd.read_csv(file,encoding='gbk')
data['maxTem'] = data['maxTem'].str.replace('℃','').astype(np.float64)
scalar = MinMaxScaler(feature_range=(0,1))
data['maxTem'] = scalar.fit_transform(data['maxTem'].values.reshape(-1,1))

datat = np.array(data['maxTem'])
datat = torch.tensor(datat)

input_dim = 1
hidden_dim = 512
num_layers = 4
output_dim = 1
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim,hidden_dim,num_layers,output_dim).to(device)
lossF = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
batch_size = 64
seq_length = 20
train_x,train_y,valid_x,valid_y = Create_Sequences_One(datat,seq_length,0.8)
train_loader,valid_loader = DataLoaders(batch_size,train_x,train_y,valid_x,valid_y)
num_epochs = 200
train_losses,valid_losses,train_mse,valid_mse,train_rmse,valid_rmse,train_mae,valid_mae = train_model(model,train_loader,valid_loader,lossF,optimizer, num_epochs, device)

plt.figure(figsize=(10,5))
plt.plot(train_losses,label="Training", color = 'red')
plt.plot(valid_losses,label="Validation", color = 'blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training and Validation')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_mse,label = 'MSE',color='blue')
plt.plot(train_rmse,label = 'RMSE',color='green')
plt.plot(valid_mse,label = 'MSE',color='orange')
plt.xlabel('Epoch')
plt.ylabel('RESULT')
plt.legend()
plt.grid(True)
plt.title('Training')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(valid_mae,label = 'MAE',color='blue')
plt.plot(valid_rmse,label = 'RMAE',color='green')
plt.plot(train_mae,label = 'MAE',color='orange')
plt.xlabel('Epoch')
plt.ylabel('RESULT')
plt.legend()
plt.grid(True)
plt.title('Validation')
plt.show()

predict_plot(model, valid_loader, scalar, device)