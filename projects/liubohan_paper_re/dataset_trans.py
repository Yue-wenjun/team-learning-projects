from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision import transforms

from study.read_data import train_dataset

BDD100k_root_dir = "C:/Users/LENOVO/PycharmProjects/PythonProject1/Road_Clean/dataset"
LABLE_dir = "C:/Users/LENOVO/PycharmProjects/PythonProject1/Road_Clean/label"

Input_Size = 224
Num_Classes = 3

def get_transform(split ='train'):
    if split == 'train':
        return transforms.Compose([transforms.Resize(Input_Size + 32),
                                        transforms.RandomCrop(Input_Size),
                                        transforms.RandomHorizontalFlip(p = 0.5),
                                        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
                                        # Imagenet的均值与标准差
    else:
        return transforms.Compose([transforms.Resize(Input_Size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])


class RWVCBDD100KDataset(Dataset):
    def __init__(self, bdd100k_root_dir, label_dir, split='train', transform=None):
        assert split in ['train', 'valid'],f"split must be 'train' or 'valid', got {split}"
        self.img_dir = os.path.join(bdd100k_root_dir, "data")
        label_json_path = os.path.join(label_dir, f"{split}_labels.json")
        if not os.path.exists(label_json_path):
            raise FileNotFoundError(f"{label_json_path} does not exist")
        with open(label_json_path, 'r') as f:
            self.labels = json.load(f)
        self.transform = transform
        self.label_map={
            "dry":0,
            "wet":1,
            "snow":2
        }
        self.label_map_weather={
            "snowy":0,
            "rainy":1,
            "clear":2,
        }
        self.label_map_visibility={
            "good":0,
            "poor":1,
        }
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label_info = self.labels[idx]
        img_filename = label_info['img']
        img_abs_path = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_abs_path).convert('RGB')
        road_label = label_info['road']
        road_label_idx = self.label_map[road_label]
        road_weather = label_info['weather']
        road_weather_idx = self.label_map_weather[road_weather]
        road_visibility = label_info['visibility']
        road_visibility_idx = self.label_map_visibility[road_visibility]
        if self.transform:
            image = self.transform(image)
        img_abs_path = img_abs_path.replace("\\", "/")
        return image, road_label_idx, road_weather_idx, road_visibility_idx


