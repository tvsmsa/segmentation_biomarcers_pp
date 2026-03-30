import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ml.biomarcers.config import Config

config = Config()

class TransUNet(nn.Module):
    """
    Transunet.
    """
    def __init__(self, img_dim=config.PATCH_SIZE, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.img_dim = img_dim
        self.num_classes = num_classes

        # CNN Encoder
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        self.patch_size = img_dim // 32
        self.hidden_dim = 768
        
        # Свертка 1x1 для проекции карт признаков CNN в нужную размерность (2048 -> 768)
        self.projection = nn.Conv2d(2048, self.hidden_dim, kernel_size=1)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=12, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        cnn_features = self.encoder(x)
        projected = self.projection(cnn_features)
        B, C, H, W = projected.shape
        projected = projected.flatten(2).transpose(1, 2)
        transformer_out = self.transformer(projected)
        transformer_out = transformer_out.transpose(1, 2).view(B, C, H, W)
        logits = self.decoder(transformer_out)
        return logits