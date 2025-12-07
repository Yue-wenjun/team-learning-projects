import os, time, tqdm
import torch
from model import UNet
from torch.utils.data import DataLoader
from dataset import MyDataset
from torch import nn, optim
from torchvision.utils import save_image


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

weight_path = 'weights/denoiser.pth'
data_path = 'data/'
save_path = 'train_image'
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.dirname(weight_path), exist_ok=True)


batch_size = 1
num_epochs = 15
lr = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MyDataset(data_path, True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    for noisy, clean in tqdm.tqdm(dataloader):
        noisy, clean = noisy.to(device), clean.to(device)
        denoised = model(noisy)

        loss = criterion(denoised, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += calculate_psnr(denoised, clean).item()

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    print(f"{time.strftime('%H:%M:%S')} Epoch [{epoch+1}/{num_epochs}]"
          f" - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
    _image = noisy[0].cpu().detach()
    _tgt_image = clean[0].cpu().detach()
    _out_image = denoised[0].cpu().detach()
    # 将图像堆叠显示：噪声图 | 原图 | 去噪结果
    img = torch.stack([_image, _tgt_image, _out_image], dim=0)
    save_image(img, os.path.join(save_path, f'epoch{epoch}.png'), normalize=True)
    
    # 保存模型
    torch.save(model.state_dict(), weight_path)
    print(f"模型已保存为 {weight_path}")
