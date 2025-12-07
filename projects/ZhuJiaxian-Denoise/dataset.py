import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
os.chdir(os.path.dirname(__file__))

def gaussianNoise(image:np.ndarray, mean, sigma) -> np.ndarray[np.float64]:
    '''给图片添加高斯噪声, 返回的图片类型为float64'''
    gauss = np.random.normal(mean,sigma,image.shape)
    noisy_img = image.astype(np.float32) + gauss
    return noisy_img.clip(0, 255)

transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path:str, train:bool, max_size:int=-1, names:list[str]=None):
        '''path: 数据集存放路径,应具有结构:
                                ├─original
                                |    └─xxxx.jpg
                                |
                                ├─trainset.txt
                                └─testset.txt
            train: 训练集或是测试集
            max_size: 限制数据集的最大数量
            names: 需要使用的图片名称,默认为None,即从txt文件读取.'''
        self.path = path
        if names is not None:
            self.names = names
        else:
            with open(os.path.join(path, 'trainset.txt' if train else 'testset.txt')) as txt:
                temp = [line.replace('\n', '') for line in txt.readlines()]
                self.names = [name for name in temp if name in os.listdir(os.path.join(self.path, 'original'))]
        if -1< max_size < len(self.names): 
            self.length = max_size
        else:
            self.length = len(self.names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        name = self.names[index]  # xx.png
        target_image = cv2.imread(os.path.join(self.path, 'original', name)).astype(np.float32)
        noisy_image = gaussianNoise(target_image, np.random.randint(-20,20), np.random.poisson(30))
        x = noisy_image/255.0
        y = target_image/255.0
        return transform(x.astype(np.float32)), transform(y.astype(np.float32))