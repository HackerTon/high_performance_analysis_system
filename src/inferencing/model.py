from enum import Enum

import torch
from torch import nn
from torchvision.models import (ResNet34_Weights, ResNet50_Weights, resnet34,
                                resnet50)
from torchvision.models.feature_extraction import create_feature_extractor


class BackboneType(Enum):
    RESNET34 = 1
    RESNET50 = 2


class MultiNet(nn.Module):
    def __init__(self, numberClass, backboneType: BackboneType):
        super().__init__()

        if backboneType == BackboneType.RESNET34:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    # "relu": "feat1",
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        elif backboneType == BackboneType.RESNET50:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = create_feature_extractor(
                backbone,
                {
                    # "relu": "feat1",
                    "layer1": "feat2",
                    "layer2": "feat3",
                    "layer3": "feat4",
                    "layer4": "feat5",
                },
            )
        else:
            raise Exception(f"No {backboneType}")

        with torch.no_grad():
            outputs_prediction = self.backbone(torch.rand([1, 3, 256, 256])).values()
            backbone_dimensions = [output.size(1) for output in outputs_prediction]

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.upsampling_2x_bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsampling_4x_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsampling_8x_bilinear = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-1],
            out_channels=256,
            kernel_size=1,
        )
        self.conv5_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv5_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv4_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-2],
            out_channels=256,
            kernel_size=1,
        )
        self.conv4_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv4_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv3_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-3],
            out_channels=256,
            kernel_size=1,
        )
        self.conv3_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv3_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )
        self.conv2_1x1 = nn.Conv2d(
            in_channels=backbone_dimensions[-4],
            out_channels=256,
            kernel_size=1,
        )
        self.conv2_3x3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.conv2_3x3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=numberClass,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        backbone_output = self.backbone(x)
        feat2, feat3, feat4, feat5 = (
            backbone_output["feat2"],
            backbone_output["feat3"],
            backbone_output["feat4"],
            backbone_output["feat5"],
        )

        conv5_mid = self.conv5_1x1(feat5).relu()
        conv5_prediction = self.conv5_3x3_1(conv5_mid).relu()
        conv5_prediction = self.conv5_3x3_2(conv5_prediction)

        conv4_lateral = self.conv4_1x1(feat4).relu()
        conv4_mid = conv4_lateral + self.upsampling_2x_bilinear(conv5_mid)
        conv4_prediction = self.conv4_3x3_1(conv4_mid).relu()
        conv4_prediction = self.conv4_3x3_2(conv4_prediction)

        conv3_lateral = self.conv3_1x1(feat3).relu()
        conv3_mid = conv3_lateral + self.upsampling_2x_bilinear(conv4_mid)
        conv3_prediction = self.conv3_3x3_1(conv3_mid).relu()
        conv3_prediction = self.conv3_3x3_2(conv3_prediction)

        conv2_lateral = self.conv2_1x1(feat2).relu()
        conv2_mid = conv2_lateral + self.upsampling_2x_bilinear(conv3_mid)
        conv2_prediction = self.conv2_3x3_1(conv2_mid).relu()
        conv2_prediction = self.conv2_3x3_2(conv2_prediction)

        final_prediction_5 = self.upsampling_8x_bilinear(conv5_prediction)
        final_prediction_4 = self.upsampling_4x_bilinear(conv4_prediction)
        final_prediction_3 = self.upsampling_2x_bilinear(conv3_prediction)
        final_prediction_2 = conv2_prediction

        return self.upsampling_4x_bilinear(
            final_prediction_5
            + final_prediction_4
            + final_prediction_3
            + final_prediction_2
        )
