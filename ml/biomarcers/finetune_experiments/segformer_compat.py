"""Align raw SegFormer state_dict names across Transformers implementations.

Renames follow Hugging Face's SegformerModel / semantic segmentation mappings:
https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py
Only names change. Tensor values/shapes remain subject to strict loading.
"""
from __future__ import annotations

import re


def modern_key(key: str) -> str:
    key = re.sub(r"^segformer\.encoder\.patch_embeddings\.(\d+)\.",
                 r"segformer.stages.\1.patch_embeddings.", key)
    key = re.sub(r"^segformer\.encoder\.block\.(\d+)\.", r"segformer.stages.\1.blocks.", key)
    key = re.sub(r"^segformer\.encoder\.layer_norm\.(\d+)\.", r"segformer.stages.\1.layer_norm.", key)
    if key.startswith("segformer.stages."):
        for old, new in (
            (".attention.self.query.", ".attention.q_proj."),
            (".attention.self.key.", ".attention.k_proj."),
            (".attention.self.value.", ".attention.v_proj."),
            (".attention.self.sr.", ".attention.sequence_reduction.sequence_reduction."),
            (".attention.self.layer_norm.", ".attention.sequence_reduction.layer_norm."),
            (".attention.output.dense.", ".attention.o_proj."),
            (".mlp.dense1.", ".mlp.fc1."),
            (".mlp.dense2.", ".mlp.fc2."),
            (".layer_norm_1.", ".layernorm_before."),
            (".layer_norm_2.", ".layernorm_after."),
        ):
            key = key.replace(old, new)
    if key.startswith("decode_head.linear_c."):
        key = key.replace("decode_head.linear_c.", "decode_head.linear_projections.", 1)
    return key


def align_segformer_state(state, target_keys):
    """Support either layout; never discard unknown or duplicate tensors."""
    targets = {}
    for key in target_keys:
        normalized = modern_key(key)
        if normalized in targets:
            raise ValueError(f"Ambiguous SegFormer model key: {key}")
        targets[normalized] = key
    result = {}
    for key, tensor in state.items():
        target = targets.get(modern_key(key), key)
        if target in result:
            raise ValueError(f"Duplicate SegFormer checkpoint key after conversion: {target}")
        result[target] = tensor
    return result
