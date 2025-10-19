import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示下载进度条

# 解决KMP冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# 1️⃣ 下载与预处理 Moving MNIST 数据（带完整性检查）
# ===============================
url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
save_path = "moving_mnist.npy"
min_valid_size = 90 * 1024 * 1024  # 最小有效文件大小（约90MB）

# 检查文件是否存在且完整，不完整则删除
if os.path.exists(save_path):
    file_size = os.path.getsize(save_path)
    if file_size < min_valid_size:
        print(f"检测到不完整文件（{file_size / 1024 / 1024:.2f}MB），删除后重新下载...")
        os.remove(save_path)

# 重新下载（如果文件不存在）
if not os.path.exists(save_path):
    print("开始下载数据集...")
    try:
        # 带进度条的下载
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=save_path) as pbar:
            def update_progress(block_num, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                pbar.update(block_size)


            urllib.request.urlretrieve(url, save_path, reporthook=update_progress)

        # 下载后再次验证
        final_size = os.path.getsize(save_path)
        if final_size < min_valid_size:
            raise Exception(f"下载文件不完整（仅{final_size / 1024 / 1024:.2f}MB）")
        print("数据集下载完成！")
    except Exception as e:
        print(f"下载失败：{e}")
        if os.path.exists(save_path):
            os.remove(save_path)  # 清除失败的文件
        exit()  # 终止程序，避免后续错误

# 加载并预处理数据
try:
    dataset = np.load(save_path)  # 正确形状应为 (20, 10000, 64, 64)
    print("Original shape:", dataset.shape)
except Exception as e:
    print(f"数据加载失败：{e}，可能文件损坏，将删除文件并退出")
    os.remove(save_path)
    exit()

# 转换维度 (samples, time, H, W)
dataset = np.swapaxes(dataset, 0, 1)
dataset = dataset[:1000, ...]  # 降低样本量
dataset = np.expand_dims(dataset, axis=-1)  # (1000, 20, 64, 64, 1)
dataset = dataset / 255.0

# 划分训练集 / 验证集
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[:int(0.9 * len(indexes))]
val_index = indexes[int(0.9 * len(indexes)):]

train_dataset = dataset[train_index]
val_dataset = dataset[val_index]


def create_shifted_frames(data):
    x = data[:, 0:data.shape[1] - 1, :, :, :]
    y = data[:, 1:data.shape[1], :, :, :]
    return x, y


x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# 转换为 tensor 并调整维度为 [B, T, C, H, W]
x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 1, 4, 2, 3)
y_train = torch.tensor(y_train, dtype=torch.float32).permute(0, 1, 4, 2, 3)
x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 1, 4, 2, 3)
y_val = torch.tensor(y_val, dtype=torch.float32).permute(0, 1, 4, 2, 3)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=2, shuffle=False)

print("Train:", x_train.shape, "Val:", x_val.shape)


# ===============================
# 2️⃣ 定义 ConvLSTM 模型
# ===============================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        i, f, o, g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 64, 64], kernel_sizes=[5, 3, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(hidden_dims)

        for i in range(self.num_layers):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(ConvLSTMCell(in_dim, hidden_dims[i], kernel_sizes[i]))

        self.conv3d = nn.Conv3d(hidden_dims[-1], 1, kernel_size=(3, 3, 3),
                                padding=(1, 1, 1))
        self.activation = nn.Sigmoid()
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(hd) for hd in hidden_dims])

    def forward(self, x):
        B, T, C, H, W = x.size()
        device = x.device

        h = [torch.zeros(B, hd, H, W).to(device) for hd in [64, 64, 64]]
        c = [torch.zeros(B, hd, H, W).to(device) for hd in [64, 64, 64]]

        outputs = []
        for t in range(T):
            x_t = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(x_t, h[i], c[i])
                h[i] = self.batchnorms[i](h[i])
                x_t = h[i]
            outputs.append(h[-1].unsqueeze(2))  # [B, hidden_dim, 1, H, W]

        y = torch.cat(outputs, dim=2)  # [B, hidden_dim, T, H, W]
        y = self.conv3d(y)  # [B, 1, T, H, W]
        y = self.activation(y)
        return y


# ===============================
# 3️⃣ 训练
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTM().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adadelta(model.parameters())

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_loss += criterion(output, y).item()

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {total_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}")

torch.save(model.state_dict(), "conv_lstm_moving_mnist.pth")
print("✅ Model saved as conv_lstm_moving_mnist.pth")

# ===============================
# 4️⃣ 可视化预测结果
# ===============================
x, y = next(iter(val_loader))
x = x.to(device)
with torch.no_grad():
    pred = model(x)[0].cpu().numpy()

fig, axes = plt.subplots(2, 5, figsize=(12, 4))
for i in range(5):
    axes[0, i].imshow(y[0, i, 0], cmap='gray')
    axes[0, i].set_title(f"True {i + 1}")
    axes[1, i].imshow(pred[0, i, 0], cmap='gray')
    axes[1, i].set_title(f"Pred {i + 1}")
    axes[0, i].axis('off')
    axes[1, i].axis('off')
plt.show()