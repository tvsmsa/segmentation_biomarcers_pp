import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from ml.biomarcers.config import Config

config = Config()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, ignore_index=config.IGNORE_INDEX):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W] с class ID
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

        # Маска валидных пикселей
        valid_mask = (targets != self.ignore_index)  # [B, H, W]

        # one-hot кодирование targets: [B, C, H, W]
        with torch.no_grad():
            targets_onehot = F.one_hot(targets.clamp(
                0, num_classes-1), num_classes).permute(0, 3, 1, 2)
            targets_onehot = targets_onehot.float() * valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask.unsqueeze(1).float()

        # TP, FP, FN
        TP = (probs * targets_onehot).sum(dim=(0, 2, 3))
        FP = (probs * (1 - targets_onehot)).sum(dim=(0, 2, 3))
        FN = ((1 - probs) * targets_onehot).sum(dim=(0, 2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FN + self.beta * FP + self.smooth)
        # исключаем класс background (0)
        tversky = tversky[1:]

        loss = 1.0 - tversky
        return loss.mean()



class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь: CrossEntropy + Focal Loss.
    Гарантированно сохраняет граф вычислений.
    """
    def __init__(self, gamma=2.0, focal_weight=2.0, smooth=1e-6, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] — выходы модели
        targets: [B, H, W] — метки классов
        """
        B, C, H, W = logits.shape

        # 1. CrossEntropy (всегда работает корректно)
        ce = self.ce_loss(logits, targets)

        # 2. Focal Loss (упрощённый, всегда сохраняет граф)
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Собираем вероятности правильных классов
        # targets: [B, H, W] → [B, 1, H, W] → [B*H*W]
        targets_flat = targets.reshape(-1)  # [B*H*W]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        # Индексы валидных пикселей
        valid_mask = (targets_flat != self.ignore_index)

        if valid_mask.sum() == 0:
            # Совсем нет валидных пикселей — rare case
            return ce + 0.0 * logits.sum()

        # Вероятности правильных классов только для валидных пикселей
        pt = probs_flat[valid_mask, targets_flat[valid_mask]]  # [N_valid]

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        # Focal loss: -weight * log(pt)
        focal_loss = -focal_weight * torch.log(pt + self.smooth)

        # Усредняем
        fl = focal_loss.mean()

        # Комбинируем
        return ce + self.focal_weight * fl