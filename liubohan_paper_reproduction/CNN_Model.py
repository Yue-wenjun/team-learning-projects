import torch
import torch.nn as nn
from torchvision import transforms, models

class RoadConditionCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.2, l2_lambda=0.1):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.top = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )
        self.backbone.fc = self.top
        self.l2_lambda = l2_lambda
    def forward(self, x):
        return self.backbone(x)
    def extract_features(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            features = torch.flatten(x, 1)
            return features
