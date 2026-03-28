import torch.nn as nn
from torchvision import models


class CNNBackbone(nn.Module):
    """
    CNN 特征提取骨干网络
    使用 ResNet18 作为特征提取器
    """

    def __init__(self, pretrained=True):
        super(CNNBackbone, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)

        # 去掉 ResNet 的最后全连接层
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # ResNet18 最终输出特征维度
        self.out_features = resnet.fc.in_features

    def forward(self, x):
        """
        :param x: 输入图像 [B, 3, 224, 224]
        :return: 特征向量 [B, out_features]
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x
