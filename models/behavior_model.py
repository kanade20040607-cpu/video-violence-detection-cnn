import torch.nn as nn
from models.cnn_backbone import CNNBackbone


class BehaviorModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = CNNBackbone(pretrained=True)
        self.classifier = nn.Linear(
            self.backbone.out_features,
            num_classes
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
