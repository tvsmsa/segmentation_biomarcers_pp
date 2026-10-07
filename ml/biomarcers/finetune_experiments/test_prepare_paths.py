"""Regression checks for local and uploaded dataset layouts."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import inspect_dataset, resolve_patch_dir


class PreparePathTests(unittest.TestCase):
    def test_supported_layouts_load_csv_with_windows_paths(self):
        for image_dir, mask_dir, prefix in (
            ("image_patches", "mask_patches", ""),
            ("images", "masks", ""),
            ("image_patches", "mask_patches", "idrid_test"),
            ("images", "masks", "nested/idrid_test"),
        ):
            with self.subTest(layout=(image_dir, mask_dir, prefix)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                name = "IDRiD_01_0_0.npy"
                images, masks = root / prefix / image_dir, root / prefix / mask_dir
                images.mkdir(parents=True)
                masks.mkdir(parents=True)
                np.save(images / name, np.zeros((512, 512, 3), dtype=np.uint8))
                np.save(masks / name, np.full((512, 512), 6, dtype=np.uint8))
                pd.DataFrame([{"image": "D:\\idrid_test\\images\\" + name,
                               "mask": "D:/idrid_test/masks/" + name}]).to_csv(
                    root / "idrid_dataset.csv", index=False)
                frame, pixels, records = inspect_dataset(root, "idrid")
                self.assertEqual(frame.iloc[0].image, (images / name).resolve().as_posix())
                self.assertEqual(frame.iloc[0]["mask"], (masks / name).resolve().as_posix())
                self.assertEqual(pixels["IDRiD_01"][6], 512 * 512)
                self.assertEqual(len(records), 1)

    def test_missing_patch_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images" / "first.npy").touch()
            with self.assertRaisesRegex(FileNotFoundError, "missing 1.*second.npy"):
                resolve_patch_dir(root, "image_patches", "image", ["first.npy", "second.npy"])

    def test_duplicate_copies_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("images", "nested/image_patches"):
                path = root / folder
                path.mkdir(parents=True)
                (path / "patch.npy").touch()
            with self.assertRaisesRegex(ValueError, "Ambiguous image"):
                resolve_patch_dir(root, "image_patches", "image", ["patch.npy"])

    def test_mask_is_not_used_as_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "masks").mkdir()
            (root / "masks" / "patch.npy").touch()
            with self.assertRaisesRegex(FileNotFoundError, "Cannot resolve image"):
                resolve_patch_dir(root, "image_patches", "image", ["patch.npy"])


if __name__ == "__main__":
    unittest.main()
