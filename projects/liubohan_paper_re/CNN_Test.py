from torch.utils.data import DataLoader
import torch
import CNN_Model
import dataset_trans
import torch.nn as nn

BATCH_SIZE = 128

label_map={
            0:"dry",
            1:"wet",
            2:"snow",
        }

BDD100k_root_dir = r"C:/Users/LENOVO/PycharmProjects/PythonProject1/Road_Clean/dataset"
LABLE_dir = r"C:/Users/LENOVO/PycharmProjects/PythonProject1/Road_Clean/label"

valid_transform = dataset_trans.get_transform(split = 'valid')

valid_dataset = dataset_trans.RWVCBDD100KDataset(
    bdd100k_root_dir=BDD100k_root_dir,
    label_dir=LABLE_dir,
    split='valid',
    transform=valid_transform
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

loss_func = nn.CrossEntropyLoss()

net = torch.load(r"C:\Users\LENOVO\PycharmProjects\PythonProject1\Road_Clean\model.pth")
net = net.cuda()
loss_test = 0
right_value = 0
for img, labels, _, __ in valid_loader:
    img = img.cuda()
    labels = labels.cuda()
    outputs = net(img)
    _, pred, = outputs.max(1)
    loss_test += loss_func(outputs, labels)
    right_value += (pred == labels).sum().item()

    images = img.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = pred.cpu().numpy()
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_data = im_data.transpose((1, 2, 0))
        im_label = labels[idx]
        im_pred = preds[idx]
        print(("预测值为{}".format(label_map[im_pred])))
        print("真实值为{}".format(label_map[im_label]))

print("准确度：{}".format(right_value/len(valid_dataset)))
