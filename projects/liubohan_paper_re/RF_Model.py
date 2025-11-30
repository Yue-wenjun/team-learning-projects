import random
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, make_scorer
from torch.utils.data import DataLoader
import joblib
from dataset_trans import RWVCBDD100KDataset, get_transform

BDD100k_root_dir = r"C:\Users\LENOVO\PycharmProjects\PythonProject1\Road_Clean\dataset"
LABLE_dir = r"C:\Users\LENOVO\PycharmProjects\PythonProject1\Road_Clean\label_1"

train_transform = get_transform(split = 'train')

cnn_model = torch.load("./model.pth")
cnn_model.eval()
cnn_model = cnn_model.cuda()

label_map={
            0:"dry",
            1:"wet",
            2:"snow",
        }

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
    transform=train_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=128,
    shuffle=False
)

def extract_cnn_features(dataset, dataloader):
    features_list = []
    labels_list = []
    weather_list = []
    vis_list = []
    with torch.no_grad():
        for images, labels, weather, vis in dataloader:
            images = images.cuda()
            features = cnn_model.extract_features(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            weather_list.append(weather.cpu().numpy())
            vis_list.append(vis.cpu().numpy())
    cnn_features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    weather_data = np.concatenate(weather_list, axis=0)
    vis_data = np.concatenate(vis_list, axis=0)
    return cnn_features, labels, weather_data, vis_data

def show_random_val_pred(num_show):
    random_idx = random.sample(range(len(val_labels)), num_show)

    for i, idx in enumerate(random_idx):
        true_label = val_labels[idx]
        pred_label = val_rf_pred[idx]
        print(label_map[true_label], label_map[pred_label])

train_cnn_features, train_labels, weather_data, vis_data = extract_cnn_features(train_dataset, train_loader)
val_cnn_features, val_labels, val_weather_data, val_vis_data = extract_cnn_features(valid_dataset, valid_loader)
train_rf_input = np.concatenate([train_cnn_features, np.expand_dims(weather_data, axis=1), np.expand_dims(vis_data, axis=1)], axis=1)
valid_rf_input = np.concatenate([val_cnn_features, np.expand_dims(val_weather_data, axis=1), np.expand_dims(val_vis_data, axis=1)], axis=1)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    class_weight='balanced',
    max_features=3,
    random_state=42,
    min_samples_leaf=1,
    bootstrap=True,
    max_samples=0.75
)

rf_model.fit(train_rf_input, train_labels)

val_rf_pred = rf_model.predict(valid_rf_input)
val_acc = accuracy_score(val_labels, val_rf_pred)
val_recall = recall_score(val_labels, val_rf_pred, average='macro')
print(f"RF验证集准确率: {val_acc:.4f}, 平均召回率: {val_recall:.4f}")
show_random_val_pred(num_show = 15)
joblib.dump(rf_model, "./rf_model.pkl")

