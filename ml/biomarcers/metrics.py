import torch
import numpy as np
from ml.biomarcers.config import Config

config = Config()

@torch.no_grad()
def dice_score_fast(preds, targets, ignore_index=config.IGNORE_INDEX):
    """
    Быстрый Dice для мультикласса без one-hot, только hard labels
    preds: logits (B, C, H, W)
    targets: (B, H, W)
    """
    preds_labels = preds.argmax(dim=1)  # (B,H,W)
    num_classes = preds.shape[1]

    dice_sum = 0.0
    count = 0
    eps = 1e-6

    for cls in range(1, num_classes):
        pred_mask = (preds_labels == cls)
        target_mask = (targets == cls)

        # игнорируем пиксели ignore_index
        valid_mask = (targets != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        TP = (pred_mask & target_mask).sum().item()
        FP = (pred_mask & (~target_mask)).sum().item()
        FN = ((~pred_mask) & target_mask).sum().item()

        if TP + FP + FN == 0:  # если класс отсутствует
            continue

        dice_cls = (2 * TP + eps) / (2 * TP + FP + FN + eps)
        dice_sum += dice_cls
        count += 1

    if count == 0:
        return 0.0
    return dice_sum / count

def compute_per_class_metrics(preds, targets, num_classes, ignore_index=255):
    """
    Метрики для каждого класса
    """
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")
    
    valid_mask = (targets != ignore_index)
    
    metrics = {
        'iou': {},
        'dice': {},
        'precision': {},
        'recall': {}
    }
    
    for class_id in range(1, num_classes):
        pred_class = (preds == class_id) & valid_mask
        target_class = (targets == class_id) & valid_mask
        
        TP = (pred_class & target_class).sum().item()
        FP = (pred_class & ~target_class).sum().item()
        FN = (~pred_class & target_class).sum().item()
        
        eps = 1e-7
        iou = TP / (TP + FP + FN + eps)
        dice = 2 * TP / (2 * TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        
        metrics['iou'][class_id] = iou
        metrics['dice'][class_id] = dice
        metrics['precision'][class_id] = precision
        metrics['recall'][class_id] = recall
    
    return metrics

def print_class_metrics(metrics, class_names, title):
    """
    Выводит метрики
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Сортируем классы по имени
    sorted_classes = sorted(class_names.items(), key=lambda x: x[1])
    
    print(f"{'Class':<30} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<10}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    
    iou_values = []
    dice_values = []
    
    for class_id, class_name in sorted_classes:
        if class_id == 0:  # пропускаем фон
            continue
            
        iou = metrics['iou'].get(class_id, 0.0)
        dice = metrics['dice'].get(class_id, 0.0)
        precision = metrics['precision'].get(class_id, 0.0)
        recall = metrics['recall'].get(class_id, 0.0)
        
        iou_values.append(iou)
        dice_values.append(dice)
        dice_str = f"{dice:<8.4f}"
        print(f"{class_name:<30} {iou:<8.4f} {dice_str:<10} {precision:<10.4f} {recall:<10.4f}")
    
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    print(f"{'MEAN':<30} {np.mean(iou_values):<8.4f} {np.mean(dice_values):<8.4f}")
    print(f"{'='*80}\n")
    
    return np.mean(dice_values)
