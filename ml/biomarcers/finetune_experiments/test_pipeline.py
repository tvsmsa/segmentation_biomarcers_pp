"""Run with: python -m unittest ml.biomarcers.finetune_experiments.test_pipeline -v"""
from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

from .common import EXPERIMENTS, experiment_frames, validate_partition
from .models import discover_group
from .prepare import filename, grouped_split, infer_source
from .scoring import metric_frame, sample_confusions, source_frame
from . import train as runner
from .training import BudgetExpired, CombinedLoss, batchnorm_for_single_image, train_epoch


class ToyDeepLab(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 3, 1)
        self.decoder = torch.nn.Conv2d(3, 3, 1)
        self.segmentation_head = torch.nn.Conv2d(3, 15, 1)

    def forward(self, x):
        return self.segmentation_head(self.decoder(self.encoder(x)))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_source_grouping_reproducibility_and_leak_rejection(self):
        frame = pd.DataFrame([{"dataset": "maples", "source_id": f"eye_{i}",
                               "image": f"image_{i}_{j}", "mask": f"mask_{i}_{j}"}
                              for i in range(10) for j in range(9)])
        train, valid = grouped_split(frame, 0.3, 42)
        self.assertEqual((len(train), len(valid)), (63, 27))
        shuffled_train = grouped_split(frame.sample(frac=1), 0.3, 42)[0]
        self.assertEqual(set(train.image), set(shuffled_train.image))
        self.assertFalse(set(train.source_id) & set(valid.source_id))
        self.assertTrue((train.groupby("source_id").size() == 9).all())
        with self.assertRaisesRegex(ValueError, "SOURCE LEAKAGE"):
            validate_partition(train, pd.concat([valid, train.head(1)]))

    def test_cross_platform_names(self):
        self.assertEqual(filename(r"D:\idrid_test\images\IDRiD_01_0_512.npy"), "IDRiD_01_0_512.npy")
        self.assertEqual(filename("D:/idrid_test/images/IDRiD_01_0_512.npy"), "IDRiD_01_0_512.npy")
        self.assertEqual(infer_source("idrid", "IDRiD_01_0_512.npy"), "IDRiD_01")
        self.assertEqual(infer_source("maples", "20051019_38557_0100_PP_1_2.npy"), "20051019_38557_0100_PP")

    def test_experiment_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ("maples", "idrid"):
                for split in ("train", "valid"):
                    pd.DataFrame([{"dataset": dataset, "source_id": f"{dataset}_{split}",
                                   "image": f"{dataset}_{split}.npy", "mask": f"{dataset}_{split}_mask.npy"}]).to_csv(
                        root / f"{dataset}_{split}.csv", index=False)
            for number, counts in ((1, (1, 1)), (2, (1, 1)), (3, (2, 2))):
                train, valid = experiment_frames(root, number)
                self.assertEqual((len(train), len(valid)), counts)
            self.assertEqual(EXPERIMENTS[1]["test"], ("maples_valid", "idrid_all"))
            self.assertEqual(EXPERIMENTS[2]["test"], ("idrid_valid", "maples_all"))

    def test_checkpoint_assignment_is_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Transunet tversky"
            folder.mkdir()
            for name in ("TransUnet_Fold3.pth", "TransUnet_Fold1_0,3027.pth", "TransUnet_Fold2_0,2997.pth"):
                (folder / name).touch()
            architecture, loss, mapping = discover_group(folder)
            self.assertEqual((architecture, loss), ("transunet", "tversky"))
            self.assertEqual(mapping[2].name, "TransUnet_Fold2_0,2997.pth")
            (folder / "extra.pth").touch()
            with self.assertRaises(ValueError):
                discover_group(folder)

    def test_metrics_false_positives_missing_classes_and_ignore(self):
        targets = torch.tensor([[[6, 6, 0, 255], [7, 0, 0, 0]]])
        predictions = torch.tensor([[[6, 0, 10, 14], [7, 0, 0, 0]]])
        matrix = sample_confusions(predictions, targets).sum(0).numpy()
        result = metric_frame(matrix).set_index("class_name")
        self.assertAlmostEqual(result.loc["hard_exudates", "dice"], 2 / 3)
        self.assertEqual(result.loc["microaneurysms", "fp"], 1)
        self.assertEqual(result.loc["microaneurysms", "dice"], 0)
        self.assertEqual(result.loc["venous_anomalies", "fp"], 0)
        self.assertEqual(len(result), 15)
        self.assertAlmostEqual(result.loc["MEAN", "dice"], (1 + 2 / 3) / 14)
        sources = source_frame({("maples", "eye"): matrix, ("idrid", "eye"): matrix})
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources.iloc[0].biomarkers_without_markup, "microaneurysms")
        self.assertTrue(pd.isna(sources.iloc[0]["drusen Dice"]))
        empty = np.zeros((15, 15), dtype=np.int64)
        self.assertEqual(metric_frame(empty).iloc[-1].dice, 0)

    def test_all_ignore_loss_is_finite_and_differentiable(self):
        for kind in ("dice", "tversky"):
            logits = torch.randn(2, 15, 4, 4, requires_grad=True)
            loss = CombinedLoss(kind)(logits, torch.full((2, 4, 4), 255))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_single_image_batchnorm_uses_running_stats_but_keeps_gradients(self):
        layer = torch.nn.BatchNorm2d(3).train()
        before = layer.running_mean.clone()
        with batchnorm_for_single_image(layer, 1):
            output = layer(torch.ones(1, 3, 1, 1))
            output.sum().backward()
            self.assertFalse(layer.training)
        self.assertTrue(layer.training)
        torch.testing.assert_close(before, layer.running_mean)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_gradient_accumulation_keeps_partial_last_group(self):
        # MSE is sample-separable: accumulated gradients must match full batches.
        images = torch.arange(15, dtype=torch.float32).reshape(5, 3, 1, 1) / 15
        targets = torch.zeros(5, 1, 1, dtype=torch.float32)
        a = torch.nn.Conv2d(3, 1, 1, bias=False)
        b = torch.nn.Conv2d(3, 1, 1, bias=False)
        b.load_state_dict(a.state_dict())
        before = a.weight.detach().clone()
        for model, batch, accumulation in ((a, 2, 2), (b, 4, 1)):
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, targets), batch_size=batch)
            loss = lambda logits, mask: (logits[:, 0] - mask).square().mean()
            train_epoch(model, loader, opt, scheduler, torch.amp.GradScaler("cuda", enabled=False),
                        loss, torch.device("cpu"), accumulation, amp=False)
        self.assertFalse(torch.equal(a.weight, before))
        torch.testing.assert_close(a.weight, b.weight)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            for split, count in (("train", 5), ("valid", 2)):
                rows = []
                for n in range(count):
                    image, mask = root / f"{split}_{n}.npy", root / f"{split}_{n}_mask.npy"
                    np.save(image, np.full((16, 16, 3), 30 + n, dtype=np.uint8))
                    target = np.zeros((16, 16), dtype=np.uint8)
                    target[:8] = 6
                    np.save(mask, target)
                    rows.append({"dataset": "maples", "source_id": f"{split}_{n}", "image": str(image), "mask": str(mask)})
                pd.DataFrame(rows).to_csv(prepared / f"maples_{split}.csv", index=False)
            initial = root / "deeplab_model_1.pth"
            torch.save(ToyDeepLab().state_dict(), initial)
            args = argparse.Namespace(epochs=2, patience=5, batch_size=2, accumulation=2, seed=42,
                    encoder_lr=1e-3, head_lr=1e-3, transunet_lr=1e-3, weight_decay=0.0, amp=False,
                    smoke_patches=0, resume=False, prepared_dir=prepared, device="cpu", num_workers=0)

            def toy_load(path, _architecture):
                data = torch.load(path, weights_only=True, map_location="cpu")
                model = ToyDeepLab()
                model.load_state_dict(data.get("model_state_dict", data))
                return model

            calls = 0

            def interrupted(*positional, **keywords):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise BudgetExpired()
                return train_epoch(*positional, **keywords)

            with patch.object(runner, "load_model", toy_load):
                runner.train_experiment(args, "deeplab", "dice", initial, 1, root / "full", {"fingerprint": "test"}, float("inf"))
                with patch.object(runner, "train_epoch", interrupted), self.assertRaises(BudgetExpired):
                    runner.train_experiment(args, "deeplab", "dice", initial, 1, root / "resumed", {"fingerprint": "test"}, float("inf"))
                args.resume = True
                runner.train_experiment(args, "deeplab", "dice", initial, 1, root / "resumed", {"fingerprint": "test"}, float("inf"))
            a = torch.load(root / "full/experiment_1/last.pth", weights_only=True)
            b = torch.load(root / "resumed/experiment_1/last.pth", weights_only=True)
            self.assertTrue(b["completed"])
            self.assertEqual(b["epoch"], 2)
            for key in a["model_state_dict"]:
                torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
            self.assertEqual(a["scheduler_state_dict"], b["scheduler_state_dict"])


if __name__ == "__main__":
    unittest.main()
