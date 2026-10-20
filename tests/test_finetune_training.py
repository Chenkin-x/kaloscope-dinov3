import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image

from dinov3.finetune.train import DEFAULTS, train


@unittest.skipIf(torch.cuda.is_available(), "Run CPU tests with CUDA_VISIBLE_DEVICES=''")
class FineTuneTrainingTests(unittest.TestCase):
    def test_ce_views_contrastive_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for artist in range(2):
                folder = root / "data" / str(artist)
                folder.mkdir(parents=True)
                for i in range(2):
                    Image.new("RGB", (40, 32), (artist * 100, i * 100, 60)).save(folder / f"{i}.png")
            for two_views, contrastive in ((False, 0.0), (True, 0.0), (False, 0.2)):
                with self.subTest(two_views=two_views, contrastive=contrastive):
                    config = deepcopy(DEFAULTS)
                    output = root / f"run_{two_views}_{contrastive}"
                    config.update(
                        model={
                            "name_or_path": None,
                            "pooling": "cls",
                            "projection_dim": 8,
                            "backbone_config": {
                                "model_type": "dinov3_vit",
                                "image_size": 32,
                                "patch_size": 16,
                                "hidden_size": 32,
                                "num_hidden_layers": 1,
                                "num_attention_heads": 2,
                                "intermediate_size": 64,
                                "num_register_tokens": 4,
                            },
                        },
                        data={"train": str(root / "data"), "val": str(root / "data"), "image_size": 32, "workers": 0},
                        output=str(output),
                        batch_size=2,
                        epochs=1,
                        two_views=two_views,
                        contrastive_weight=contrastive,
                        precision="fp32",
                    )
                    train(config)
                    progress = json.loads((output / "steps.jsonl").read_text().splitlines()[0])
                    self.assertEqual(progress["views_per_original"], 2 if two_views or contrastive else 1)
                    self.assertEqual(progress["original_images_per_global_batch"], 2)
                    config.update(resume=str(output / "last.pt"), epochs=2)
                    train(config)
                    checkpoint = torch.load(output / "last.pt", weights_only=False)
                    self.assertEqual(checkpoint["epoch"], 1)
                    self.assertEqual(checkpoint["global_step"], 4)
                    self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint["model"].values()))


if __name__ == "__main__":
    unittest.main()
