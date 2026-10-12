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
        resnet_layers = list(resnet.children())
        self.encoder_stage0 = nn.Sequential(*resnet_layers[:4])   # -> [B, 64, H/4, W/4]
        self.encoder_stage1 = resnet_layers[4]                     # -> [B, 256, H/4, W/4]
        self.encoder_stage2 = resnet_layers[5]                     # -> [B, 512, H/8, W/8]
        self.encoder_stage3 = resnet_layers[6]                     # -> [B, 1024, H/16, W/16]
        self.encoder_stage4 = resnet_layers[7]                     # -> [B, 2048, H/32, W/32]

        self.patch_size = img_dim // 32
        self.hidden_dim = 768

        # Свертка 1x1 для проекции карт признаков CNN в нужную размерность (2048 -> 768)
        self.projection = nn.Conv2d(2048, self.hidden_dim, kernel_size=1)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_size * self.patch_size, self.hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=12, dim_feedforward=self.hidden_dim * 4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Decoder
        self.decoder_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Блок 2: 512 + skip от stage2 (512)
        self.decoder_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Блок 3: 256 + skip от stage1 (256)
        self.decoder_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Финальный блок: upsampling до полного разрешения
        self.decoder_up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        f0 = self.encoder_stage0(x)    # [B, 64, H/4, W/4]
        f1 = self.encoder_stage1(f0)   # [B, 256, H/4, W/4]   <- skip
        f2 = self.encoder_stage2(f1)   # [B, 512, H/8, W/8]   <- skip
        f3 = self.encoder_stage3(f2)   # [B, 1024, H/16, W/16] <- skip
        f4 = self.encoder_stage4(f3)   # [B, 2048, H/32, W/32]

        # ==================== TRANSFORMER ====================
        projected = self.projection(f4)  # [B, 768, H', W']
        B, C, H, W = projected.shape

        # [B, 768, H', W'] -> [B, N, 768]
        projected = projected.flatten(2).transpose(1, 2)

        # Добавляем позиционное кодирование
        projected = projected + self.pos_embedding[:, :projected.size(1), :]

        # Transformer
        transformer_out = self.transformer(projected)

        # [B, N, 768] -> [B, 768, H', W']
        transformer_out = transformer_out.transpose(1, 2).view(B, C, H, W)

        # ==================== DECODER (со skip connections) ====================
        # Блок 1: upsample + concat с f3
        d = self.decoder_up1(transformer_out)  # [B, 768, H/16, W/16]
        d = self._match_size(d, f3)
        d = torch.cat([d, f3], dim=1)          # [B, 768+1024, H/16, W/16]
        d = self.decoder_conv1(d)              # [B, 512, H/16, W/16]

        # Блок 2: upsample + concat с f2
        d = self.decoder_up2(d)                # [B, 512, H/8, W/8]
        d = self._match_size(d, f2)
        d = torch.cat([d, f2], dim=1)          # [B, 512+512, H/8, W/8]
        d = self.decoder_conv2(d)              # [B, 256, H/8, W/8]

        # Блок 3: upsample + concat с f1
        d = self.decoder_up3(d)                # [B, 256, H/4, W/4]
        d = self._match_size(d, f1)
        d = torch.cat([d, f1], dim=1)          # [B, 256+256, H/4, W/4]
        d = self.decoder_conv3(d)              # [B, 128, H/4, W/4]

        # Финальный upsample до полного разрешения
        d = self.decoder_up4(d)                # [B, 128, H, W]
        logits = self.final_conv(d)            # [B, num_classes, H, W]

        return logits

    @staticmethod
    def _match_size(src, ref):
        """
        Выравнивает пространственные размеры src под ref через билинейную интерполяцию.
        Нужно на случай, если размеры не совпадают из-за нечётных размеров входа.
        """
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(
                src, size=ref.shape[-2:], mode='bilinear', align_corners=True
            )
        return src