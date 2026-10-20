import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from dinov3.data import Originals, load_metadata
from dinov3.evaluate import content_mask, load_panel
from dinov3.model import Encoder
from dinov3.sampling import PairPool
from train_temporal import train
from tests.temporal_fixture import make_fixture


class TemporalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_center_crop_and_split_guard(self):
        dataset = Originals(self.config["paths"], 32)
        image, index = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertEqual(index, 0)
        meta = load_metadata(self.config["metadata"], len(dataset))
        load_panel(self.config["panel"], meta)
        # Test images must never enter validation, even with a misleading report.
        np.save(self.root / "data/query.npy", np.array([7, 15]))
        with self.assertRaisesRegex(ValueError, "test is sealed"):
            load_panel(self.config["panel"], meta)
        families = np.arange(16)
        families[6] = families[0]
        np.save(self.root / "data/work_family_id.npy", families)
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_metadata(self.config["metadata"], len(dataset))

    def test_large_batch_and_no_positive_guard(self):
        pool = PairPool(self.config["metadata"])
        batch = next(pool.batches(0, 42, 128))
        self.assertEqual(batch.shape, (128,))
        self.assertTrue(set(pool.indices) <= set(batch[::2]))
        np.save(self.root / "data/work_family_id.npy", np.zeros(16, dtype="int64"))
        with self.assertRaisesRegex(ValueError, "No distinct-work"):
            PairPool(self.config["metadata"])

    def test_cross_content_requires_known_disjoint_tags(self):
        content = {"0": ["a", "p"], "1": ["b", "q"], "2": ["a", "r"], "3": ["", "q"]}
        allowed = content_mask(np.array([0]), np.array([1, 2, 3]), content)
        np.testing.assert_array_equal(allowed, [[True, False, False]])

    def test_distributed_loss_matches_single_process_gradients(self):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "tests.distributed_loss_worker",
        ]
        env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "true",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        }
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipIf(torch.cuda.is_available(), "Run CPU tests with CUDA_VISIBLE_DEVICES='' to protect active training")
    def test_exact_resume_and_checkpoint_features(self):
        continuous = deepcopy(self.config)
        continuous["output"] = str(self.root / "continuous")
        train(continuous, max_steps=3)
        train(self.config, max_steps=1)
        train(self.config, resume=True, max_steps=3)
        first = torch.load(Path(continuous["output"]) / "last.pt", weights_only=False)
        resumed = torch.load(Path(self.config["output"]) / "last.pt", weights_only=False)
        self.assertEqual((resumed["step"], resumed["epoch"], resumed["offset"]), (3, 0, 3))
        for key in first["model"]:
            torch.testing.assert_close(first["model"][key], resumed["model"][key], rtol=0, atol=0)
        self.assertEqual(first["controller"], resumed["controller"])
        # Loading trained weights does not require the initial pretrained file.
        model_config = {**resumed["model_config"], "name_or_path": "/nonexistent/pretrained"}
        model = Encoder(model_config, initialize=False).eval()
        model.load_state_dict(resumed["model"])
        with torch.no_grad():
            features = model(Originals(self.config["paths"], 32)[0][0][None])
        self.assertEqual(features.shape, (1, 80))
        self.assertTrue(torch.isfinite(features).all())
        changed = deepcopy(self.config)
        changed["window_days"] = 10
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            train(changed, resume=True)
        days = np.load(self.root / "data/days.npy")
        days[0] += 1
        np.save(self.root / "data/days.npy", days)
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            train(self.config, resume=True)

    def test_torchrun_two_cpu_ranks_and_resume(self):
        config = deepcopy(self.config)
        config["batch_size"] = 4
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "train_temporal",
            "--config",
            str(path),
        ]
        env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "true",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        }
        for args in (["--max-steps", "1"], ["--resume", "--max-steps", "2"]):
            process = subprocess.run(command + args, env=env, capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        checkpoint = torch.load(Path(config["output"]) / "last.pt", weights_only=False)
        self.assertEqual(checkpoint["step"], 2)
        state = Path(config["output"]) / checkpoint["accelerate_state"]
        self.assertEqual(len(list(state.glob("random_states_*.pkl"))), 2)
        self.assertTrue(all(torch.isfinite(value).all() for value in checkpoint["model"].values()))
        validation = [
            json.loads(line) for line in (Path(config["output"]) / "validation.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["step"] for row in validation], [0, 2])
        self.assertTrue(all(not row["metrics"]["test_used"] for row in validation))


if __name__ == "__main__":
    unittest.main()
