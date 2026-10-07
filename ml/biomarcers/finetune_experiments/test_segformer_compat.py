"""Run under both legacy and modern Transformers installations."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from .models import forward_logits, load_model
from .segformer_compat import align_segformer_state, modern_key
from .training import make_optimizer


class SegformerCompatibilityTests(unittest.TestCase):
    def test_conversion_matches_reported_kaggle_names(self):
        examples = {
            "segformer.encoder.patch_embeddings.0.proj.weight": "segformer.stages.0.patch_embeddings.proj.weight",
            "segformer.encoder.block.1.2.attention.self.query.bias": "segformer.stages.1.blocks.2.attention.q_proj.bias",
            "segformer.encoder.block.1.2.attention.self.sr.weight": "segformer.stages.1.blocks.2.attention.sequence_reduction.sequence_reduction.weight",
            "segformer.encoder.block.1.2.attention.self.layer_norm.bias": "segformer.stages.1.blocks.2.attention.sequence_reduction.layer_norm.bias",
            "segformer.encoder.block.1.2.attention.output.dense.weight": "segformer.stages.1.blocks.2.attention.o_proj.weight",
            "segformer.encoder.block.1.2.layer_norm_2.weight": "segformer.stages.1.blocks.2.layernorm_after.weight",
            "segformer.encoder.block.1.2.mlp.dense1.weight": "segformer.stages.1.blocks.2.mlp.fc1.weight",
            "segformer.encoder.layer_norm.3.weight": "segformer.stages.3.layer_norm.weight",
            "decode_head.linear_c.3.proj.bias": "decode_head.linear_projections.3.proj.bias",
        }
        for old, new in examples.items():
            with self.subTest(key=old):
                tensor = object()
                self.assertEqual(modern_key(old), new)
                self.assertIs(align_segformer_state({old: tensor}, [new])[new], tensor)
                self.assertIs(align_segformer_state({new: tensor}, [old])[old], tensor)

    def test_duplicate_and_unknown_parameters(self):
        old = "segformer.encoder.layer_norm.0.weight"
        new = "segformer.stages.0.layer_norm.weight"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            align_segformer_state({old: object(), new: object()}, [new])
        self.assertIn("unknown", align_segformer_state({"unknown": object()}, [new]))

    def test_real_model_strict_load_training_and_reload(self):
        torch.set_num_threads(2)
        config = SegformerConfig(num_labels=15, depths=[1, 1, 1, 1],
                                 hidden_sizes=[8, 16, 32, 64], num_attention_heads=[1, 2, 4, 8],
                                 decoder_hidden_size=16)
        model = SegformerForSemanticSegmentation(config)
        original = model.state_dict()
        modern = {modern_key(k): v for k, v in original.items()}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "ml.biomarcers.finetune_experiments.models.build_model",
            side_effect=lambda _: SegformerForSemanticSegmentation(config),
        ):
            path = Path(tmp) / "checkpoint.pth"
            torch.save(modern, path)
            loaded = load_model(path, "segformer")
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            optimizer = make_optimizer(loaded, "segformer", 1e-5, 5e-4, 1e-4)
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(parameters), len({id(p) for p in parameters}))
            self.assertEqual({id(p) for p in parameters}, {id(p) for p in loaded.parameters()})
            images = torch.randn(2, 3, 32, 32)
            before = loaded.decode_head.classifier.weight.detach().clone()
            loss = forward_logits(loaded.train(), images, (32, 32)).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
            optimizer.step()
            self.assertFalse(torch.equal(before, loaded.decode_head.classifier.weight))
            loaded.eval()
            with torch.no_grad():
                expected = forward_logits(loaded, images, (32, 32))
                torch.save({"model_state_dict": loaded.state_dict()}, path)
                restored = load_model(path, "segformer").eval()
                torch.testing.assert_close(forward_logits(restored, images, (32, 32)), expected)
            broken = dict(modern)
            broken.pop(next(iter(broken)))
            torch.save(broken, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_model(path, "segformer")


if __name__ == "__main__":
    unittest.main()
