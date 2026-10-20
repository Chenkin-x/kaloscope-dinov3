import unittest

from PIL import Image

from dinov3.data import prepare_rgb


class TrainingDataTests(unittest.TestCase):
    def test_transparency_and_orientation(self):
        image = Image.new("RGBA", (12, 20), (255, 0, 0, 0))
        image.getexif()[274] = 6
        converted = prepare_rgb(image)
        self.assertEqual(converted.size, (20, 12))
        self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
