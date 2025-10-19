import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import UNet
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/best_model.pth'
data_path = 'data'
save_path = 'train_image'
if not os.path.exists(save_path): os.mkdir(save_path)
# 数据加载
data_loader = DataLoader(MyDataset(data_path, train=True), batch_size=4, shuffle=True)
net = UNet().to(device)  # 移除num_classes参数
if os.path.exists(weight_path): net.load_state_dict(torch.load(weight_path, device))

# 选择适合的损失函数
# 方案1: L1损失 - 对异常值更鲁棒，边缘保持更好
loss_fun = nn.L1Loss()

# 方案2: MSE损失 - 训练更稳定，PSNR更高
# loss_fun = nn.MSELoss()


optimizer = optim.Adam(net.parameters(), lr=1e-4)

epoch = 1
while epoch < 200:
    for i, (image, tgt_image) in enumerate(tqdm.tqdm(data_loader)):
        image, tgt_image = image.to(device), tgt_image.to(device)
        out_image = net(image)
        
        # 确保目标图像是浮点型且在正确范围内
        if tgt_image.max() > 1.0:
            tgt_image = tgt_image.float() / 255.0
        
        train_loss = loss_fun(out_image, tgt_image)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


        # 可视化 - 每100个batch保存一次
        if i % 100 == 0:
            _image = image[0].cpu().detach()
            _tgt_image = tgt_image[0].cpu().detach()
            _out_image = out_image[0].cpu().detach()
            
            # 将图像堆叠显示：噪声图 | 原图 | 去噪结果
            img = torch.stack([_image, _tgt_image, _out_image], dim=0)
            save_image(img, f'{save_path}/epoch{epoch}_batch{i}.png', normalize=True)
    
    if epoch % 10 == 0:  # 每10个epoch保存一次
        torch.save(net.state_dict(), weight_path)
        print('save successfully!')
    
    epoch += 1
