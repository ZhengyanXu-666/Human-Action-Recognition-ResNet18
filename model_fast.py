# model_fast.py
import torch.nn as nn
import torchvision.models as models


class FastResNet18(nn.Module):


    def __init__(self, num_classes=30):
        super().__init__()

        
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        
        self.base.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base(x)
