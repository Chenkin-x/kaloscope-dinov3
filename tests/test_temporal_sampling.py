import tempfile
import unittest
from pathlib import Path

import numpy as np

from dinov3.control import Convergence
from dinov3.sampling import PairPool


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        arrays = {
            "artist_ids": [1, 1, 1, 1, 2, 2, 3, 4, 4],
            "days": [0, 365, 366, 500, 100, 102, 0, 0, 100],
            "work_family_id": [0, 1, 2, 2, 4, 5, 6, 7, 8],
            "group_ids": [0, 1, 1, 1, 2, 2, 3, 4, 4],
            "split": [0, 0, 0, 0, 0, 0, 0, 1, 2],
        }
        for k, v in arrays.items():
            np.save(self.root / f"{k}.npy", np.array(v))

    def tearDown(self):
        self.temp.cleanup()

    def test_temporal_mates_and_family_exclusion(self):
        p = PairPool(self.root)
        rng = np.random.default_rng(8)
        for row in p.indices:
            for _ in range(50):
                mate, positive = p.mate(row, rng)
                self.assertIn(mate, p.indices)
                if positive:
                    self.assertEqual(p.artist[row], p.artist[mate])
                    self.assertLessEqual(abs(p.day[row] - p.day[mate]), 365)
                    self.assertNotEqual(p.family[row], p.family[mate])
                else:
                    self.assertNotEqual(p.artist[row], p.artist[mate])
        self.assertEqual(p.report["no_temporal_mate_images"], 1)

    def test_complete_coverage_and_repeatability(self):
        p = PairPool(self.root)
        a = np.array(list(p.batches(0, 42, 8)))
        b = np.array(list(p.batches(0, 42, 8)))
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(set(a[:, ::2].reshape(-1)), set(p.indices))
        self.assertTrue(set(a.reshape(-1)) <= set(p.indices))

    def test_no_early_plateau_before_full_coverage(self):
        c = Convergence()
        for _ in range(30):
            r = c.update(0.5, 0.5, True)
        self.assertEqual(c.factor, 1.0)
        self.assertIsNone(r["stop_reason"])

    def test_plateau_and_regression_are_distinct(self):
        c = Convergence()
        c.update(0.5, 0, True)
        for _ in range(18):
            r = c.update(0.5, 2, True)
        self.assertEqual(r["stop_reason"], "validation_plateau")
        c = Convergence()
        c.update(0.5, 0, True)
        for _ in range(3):
            r = c.update(0.4, 0.1, True)
        self.assertEqual(r["stop_reason"], "validation_regression")


if __name__ == "__main__":
    unittest.main()
