import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class NeuralNetwork(nn.Module):
    def __init__(self, numberClass):
        _resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(_resnet50.children())[:-1])
        self.flattten = nn.Flatten(2)
        self.final_layer = nn.Linear(2048, numberClass)

    def forward(self, x):
        return self.final_layer(torch.squeeze(self.flattten(self.backbone(x)), dim=-1))