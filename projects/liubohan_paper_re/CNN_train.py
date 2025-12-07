import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import CNN_Model
import dataset_trans
from dataset_trans import RWVCBDD100KDataset
from dataset_trans import get_transform




BATCH_SIZE = 128
EPOCHS = 50

SAVE_DIR = "./model_save"
BDD100k_root_dir = "C:/Users/LENOVO/PycharmProjects/PythonProject1/Road_Clean/dataset"
LABLE_dir = "C:/Users/LENOVO/PycharmProjects/PythonProject1/Road_Clean/label"


train_transform = get_transform(split = 'train')
valid_transform = get_transform(split = 'valid')

train_dataset = RWVCBDD100KDataset(
    bdd100k_root_dir=BDD100k_root_dir,
    label_dir=LABLE_dir,
    split='train',
    transform=train_transform
)
valid_dataset = RWVCBDD100KDataset(
    bdd100k_root_dir=BDD100k_root_dir,
    label_dir=LABLE_dir,
    split='valid',
    transform=valid_transform
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


model = CNN_Model.RoadConditionCNN()
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    params=model.backbone.fc.parameters(),
    lr=0.01,
    momentum=0.25,
    weight_decay=1e-4,
)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (imgs, labels, weather_labels, visibility_labels) in enumerate(loader):
        imgs = imgs.cuda()
        labels = labels.cuda()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += torch.sum(preds == labels).item()
        if (batch_idx + 1) % 10 == 0:
            batch_acc = 100 * correct / total

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * correct / total
    print(f"Epoch[{epoch+1}/{EPOCHS}]: Batch Accuracy: {avg_acc:.3f},loss: {avg_loss:.3f}")
    return avg_loss,avg_acc

best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

torch.save(model, "./model.pth")
