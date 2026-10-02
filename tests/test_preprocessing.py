import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from dinov3.finetune.data import FolderViews, image_transform, prepare_rgb


class TrainingDataTests(unittest.TestCase):
    def test_transparency_and_orientation(self):
        image = Image.new("RGBA", (12, 20), (255, 0, 0, 0))
        image.getexif()[274] = 6
        converted = prepare_rgb(image)
        self.assertEqual(converted.size, (20, 12))
        self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))

    def test_flip_only_keeps_batch_shape(self):
        transform = image_transform(32, augment={"random_horizontal_flip": True})
        images = [Image.new("RGB", size) for size in [(30, 80), (90, 40)]]
        self.assertEqual(torch.stack([transform(i) for i in images]).shape, (2, 3, 32, 32))

    def test_eval_is_deterministic(self):
        transform = image_transform(32)
        image = Image.new("RGBA", (72, 48), (80, 150, 30, 128))
        self.assertTrue(torch.equal(transform(image), transform(image)))

    def test_folder_loader_preserves_alpha_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "artist"
            p.mkdir()
            Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(p / "a.png")
            dataset = FolderViews(directory, lambda image: image)
            self.assertEqual(dataset[0][0].getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
