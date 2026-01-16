"""
Sketch Recognition Model
使用预训练ResNet进行草图分类
"""

import torch
import torch.nn as nn
from torchvision import models


class SketchRecognitionModel(nn.Module):
    """基于ResNet的草图识别模型"""
    
    def __init__(self, num_classes, backbone='resnet18', pretrained=True, dropout=0.5):
        super().__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # 替换分类层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    model = SketchRecognitionModel(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
