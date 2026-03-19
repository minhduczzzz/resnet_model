import torch.nn as nn
from torchvision import models


class DogBreedResNet(nn.Module):
    def __init__(self, num_classes=120, pretrained=True):
        super().__init__()

        # Load ResNet18 pretrained
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.resnet = models.resnet18(weights=weights)

        # Lấy số input của FC
        in_features = self.resnet.fc.in_features

        # 🔥 Thay FC bằng head mới (có Dropout)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)