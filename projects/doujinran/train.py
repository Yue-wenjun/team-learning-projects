import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset import get_data_loaders
from model import CSPricePredictor

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 超参数
time_step = 30  # 时间步长
batch_size = 32  # 批次大小
learning_rate = 0.001  # 学习率
num_epochs = 100  # 训练轮数
device = try_gpu()

# 1. 数据准备
print("加载数据...")
train_loader, val_loader, test_loader, feature_names = get_data_loaders(
    csv_file='csgo_prices.csv',
    time_step=time_step,
    batch_size=batch_size
)

# 从训练数据中获取输入特征数量
sample_batch, _ = next(iter(train_loader))
input_size = sample_batch.shape[2]  # 特征数量 [batch_size, time_step, features]

# 模型参数
hidden_size = 128
num_layers = 3
output_size = 1

print(f"输入特征数: {input_size}")
print(f"时间步长: {time_step}")
print(f"使用设备: {device}")

# 初始化模型
model = CSPricePredictor(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_Y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1

    # 验证
    model.eval()
    val_loss = 0
    val_count = 0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_Y)
            val_loss += loss.item()
            val_count += 1

    avg_train_loss = epoch_loss / batch_count
    avg_val_loss = val_loss / val_count

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# 保存模型
import os
os.makedirs('./model_param', exist_ok=True)
torch.save(model.state_dict(), './model_param/cs_model.pth')
print("模型已保存到 ./model_param/cs_model.pth")

# 测试模型
model.eval()
test_loss = 0
test_count = 0
with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_Y)
        test_loss += loss.item()
        test_count += 1

avg_test_loss = test_loss / test_count
print(f'测试损失: {avg_test_loss:.4f}')