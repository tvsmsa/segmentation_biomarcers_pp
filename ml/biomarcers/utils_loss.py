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


class FocalLoss(nn.Module):
    """
    Focal Loss: динамически увеличивает вес ошибок на трудно-классифицируемых пикселях
    """
    def __init__(self, gamma=2.0, smooth=1e-6, ignore_index=config.IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] — сырые выходы модели
        targets: [B, H, W] — метки классов
        """
        num_classes = logits.shape[1]

        # Вероятности классов
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

        # Маска валидных пикселей (исключаем ignore_index)
        valid_mask = (targets != self.ignore_index)  # [B, H, W]

        # One-hot метки
        with torch.no_grad():
            targets_onehot = F.one_hot(
                targets.clamp(0, num_classes - 1),
                num_classes
            ).permute(0, 3, 1, 2).float()  # [B, C, H, W]
            targets_onehot = targets_onehot * valid_mask.unsqueeze(1).float()

        # Применяем маску к вероятностям
        probs = probs * valid_mask.unsqueeze(1).float()

        # Собираем вероятности истинного класса: pt [B, 1, H, W]
        pt = (probs * targets_onehot).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        focal_weight = (1.0 - pt) ** self.gamma

        # Focal loss для всех пикселей
        focal_loss = -focal_weight * torch.log(pt + self.smooth)  # [B, 1, H, W]

        # Усредняем только по foreground (классы != 0) и валидным пикселям
        foreground_mask = (targets != 0) & valid_mask  # [B, H, W]

        if foreground_mask.sum() > 0:
            # Выбираем пиксели foreground и усредняем
            loss = focal_loss.squeeze(1)[foreground_mask].mean()
        else:
            # Если в батче нет foreground, возвращаем 0
            loss = focal_loss.mean() * 0.0 + 0.0 * logits.mean()

        return loss