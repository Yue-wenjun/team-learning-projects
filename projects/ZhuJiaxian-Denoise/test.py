import os
import cv2
import numpy as np
import torch
from dataset import MyDataset, gaussianNoise, transform
from model import UNet
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet()
weights='weights/denoiser.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights, map_location = device))
else:
    print('no loading')

net.eval()
test_iter = DataLoader(MyDataset("data/", train=False), batch_size=1, shuffle=True)
for i, (image, tgt) in enumerate(test_iter):
    out=net(image)

    result = np.squeeze(out.detach().numpy() * 255)
    # print(result.shape) # CHW: (3, 160, 160)
    padding = np.zeros((3,160,5))
    show = np.concatenate((tgt.squeeze()*255, padding, image.squeeze()*255, padding, result), axis=2)
    cv2.imshow('result', show.transpose((1,2,0)).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if i>10: break

# image_path = "data/original/1089.jpg"
# image_path = "color_map.jpg"
# image = cv2.imread(image_path)

# transf = lambda im: torch.unsqueeze(transform(im.astype(np.float32)), dim=0)
# noisy = gaussianNoise(image, np.random.randint(-20,20), np.random.poisson(30))
# image, noisy = transf(image/255.0), transf(noisy/255.0)
# out=net(image)
# padding = np.zeros((3,160,5))
# out_image = out.detach().numpy() * 255
# show = np.concatenate((noisy.squeeze()*255, padding, image.squeeze()*255, padding, out_image.squeeze()), axis=2)
# cv2.imshow('result', show.transpose((1,2,0)).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()