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


    metrics = {
        'iou': {},
        'dice': {},
        'precision': {},
        'recall': {},
        'count': {}
    }

    preds_1 = preds.shape[0]

    # Списки метрик по изображениям для каждого класса
    per_class_iou = {c: [] for c in range(1, num_classes)}
    per_class_dice = {c: [] for c in range(1, num_classes)}
    per_class_precision = {c: [] for c in range(1, num_classes)}
    per_class_recall = {c: [] for c in range(1, num_classes)}

    for b in range(preds_1):
        pred_b = preds[b]
        target_b = targets[b]
        valid_mask = (target_b != ignore_index)

        for class_id in range(1, num_classes):
            pred_class = (pred_b == class_id) & valid_mask
            target_class = (target_b == class_id) & valid_mask

            TP = (pred_class & target_class).sum().item()
            FP = (pred_class & ~target_class).sum().item()
            FN = (~pred_class & target_class).sum().item()

            # Пропускаем изображения, где класса нет в GT (нет ни одного пикселя GT)
            if target_class.sum().item() == 0:
                continue

            eps = 1e-7
            iou = TP / (TP + FP + FN + eps)
            dice = 2 * TP / (2 * TP + FP + FN + eps)
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)

            per_class_iou[class_id].append(iou)
            per_class_dice[class_id].append(dice)
            per_class_precision[class_id].append(precision)
            per_class_recall[class_id].append(recall)

    for class_id in range(1, num_classes):
        n = len(per_class_iou[class_id])
        metrics['count'][class_id] = n
        metrics['iou'][class_id] = float(np.mean(per_class_iou[class_id])) if per_class_iou[class_id] else 0.0
        metrics['dice'][class_id] = float(np.mean(per_class_dice[class_id])) if per_class_dice[class_id] else 0.0
        metrics['precision'][class_id] = float(np.mean(per_class_precision[class_id])) if per_class_precision[class_id] else 0.0
        metrics['recall'][class_id] = float(np.mean(per_class_recall[class_id])) if per_class_recall[class_id] else 0.0

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

    print(f"{'Class':<30} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<10} {'N':<6}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")

    iou_values = []
    dice_values = []

    for class_id, class_name in sorted_classes:
        if class_id == 0:  # пропускаем фон
            continue

        n = metrics.get('count', {}).get(class_id, None)
        if n is not None and n == 0:
            # класса нет в датасете — не учитываем в среднем
            print(f"{class_name:<30} {'—':<8} {'—':<8} {'—':<10} {'—':<10} {n:<6}")
            continue

        iou = metrics['iou'].get(class_id, 0.0)
        dice = metrics['dice'].get(class_id, 0.0)
        precision = metrics['precision'].get(class_id, 0.0)
        recall = metrics['recall'].get(class_id, 0.0)
        
        iou_values.append(iou)
        dice_values.append(dice)
        print(f"{class_name:<30} {iou:<8.4f} {dice:<8.4f} {precision:<10.4f} {recall:<10.4f} {n if n is not None else '-':<6}")

    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
    mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
    print(f"{'MEAN':<30} {mean_iou:<8.4f} {mean_dice:<8.4f}")
    print(f"{'='*80}\n")

    return mean_dice
