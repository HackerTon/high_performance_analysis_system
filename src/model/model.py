import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class UNETNetwork(nn.Module):
    def __init__(self, numberClass):
        super().__init__()
        _resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = create_feature_extractor(
            _resnet50,
            {
                "relu": "feat1",
                "layer1": "feat2",
                "layer2": "feat3",
                "layer3": "feat4",
                "layer4": "feat5",
            },
        )
        self.upsample2x1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2x2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2x3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2x4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2x5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv5 = nn.Conv2d(
            in_channels=2048,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            in_channels=1280,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(
            in_channels=768,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(
            in_channels=320,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        self.relu9 = nn.ReLU()
        self.convfinal = nn.Conv2d(
            in_channels=256,
            out_channels=numberClass,
            kernel_size=1,
        )
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat1, feat2, feat3, feat4, feat5 = (
            backbone_output["feat1"],
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )
        feat4to6 = self.upsample2x1(self.relu5(self.conv5(feat5)))
        feat3to7 = self.upsample2x2(
            self.relu6(self.conv6(torch.concat([feat4, feat4to6], dim=1)))
        )
        feat2to8 = self.upsample2x3(
            self.relu7(self.conv7(torch.concat([feat3, feat3to7], dim=1)))
        )
        feat1to9 = self.upsample2x4(
            self.relu8(self.conv8(torch.concat([feat2, feat2to8], dim=1)))
        )
        featout = self.upsample2x5(
            self.relu9(self.conv9(torch.concat([feat1, feat1to9], dim=1)))
        )
        return self.final_activation(self.convfinal(featout))