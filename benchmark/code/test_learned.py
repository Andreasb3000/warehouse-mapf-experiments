"""Contract and causality tests for learned forecast baselines."""
import unittest
import contextlib
import io
from pathlib import Path
import tempfile
import numpy as np
import torch

from learned import SpatialPredictor, loss_components, split_indices, validate_arrays, train_one


class LearnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        n, h, w = 3, 7, 9
        static = np.ones((n, 3, h, w), dtype=np.float32)
        static[:, :, 0] = 0
        road = static[0, 0].copy()
        return {"history": np.zeros((n, 6, 2, h, w), dtype=np.float32),
                "plan": np.zeros((n, 3, 2, h, w), dtype=np.float32),
                "static": static, "target": np.zeros((n, 3, 2, h, w), dtype=np.float32),
                "road": road, "roi": road.copy()}

    def test_contract_and_nonnegative_conservation(self):
        data = self.data()
        self.assertEqual(validate_arrays(data), 3)
        for residual in (False, True):
            model = SpatialPredictor(residual=residual)
            args = [torch.from_numpy(data[k]) for k in ("history", "plan", "static")]
            result = model(*args)
            self.assertEqual(tuple(result.shape), (3, 3, 2, 7, 9))
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue((result >= 0).all())
            self.assertTrue((result[:, :, 1] <= result[:, :, 0]).all())
            self.assertEqual(float(result[:, :, :, 0].sum().detach()), 0)
            loss, _ = loss_components(result, torch.from_numpy(data["target"]),
                                      torch.from_numpy(data["road"]), torch.from_numpy(data["roi"]))
            loss.backward()
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_model_cannot_read_target_or_modify_inputs(self):
        data = self.data()
        args = [torch.from_numpy(data[k]) for k in ("history", "plan", "static")]
        original = [a.clone() for a in args]
        model = SpatialPredictor().eval()
        with torch.inference_mode():
            first = model(*args)
            data["target"][:] = 10000
            second = model(*args)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(args, original)))

    def test_split_workload_disjointness(self):
        records = [{"split": split, "workload_id": i} for i, split in enumerate(
            ("train", "val", "test"))]
        result = split_indices({"samples": records}, 3)
        self.assertEqual(result["validation"].tolist(), [1])
        records[2]["workload_id"] = 0
        with self.assertRaisesRegex(ValueError, "multiple data splits"):
            split_indices({"samples": records}, 3)
        records = [{"split": s, "workload_seed": i} for i, s in enumerate(
            ("train", "validation", "test"))]
        records[2]["workload_seed"] = 0
        with self.assertRaisesRegex(ValueError, "multiple data splits"):
            split_indices({"samples": records}, 3)

    def test_bad_targets_rejected(self):
        data = self.data()
        data["target"][0, 0, 1, 3, 3] = 1
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_arrays(data)

    def test_residual_zero_head_recovers_plan(self):
        data = self.data()
        data["plan"][:, :, 0, 4, 4] = 1.2
        model = SpatialPredictor(residual=True)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            result = model(*[torch.from_numpy(data[k]) for k in ("history", "plan", "static")])
        expected = (torch.from_numpy(data["plan"][:, :, 0]) + .005)
        expected *= torch.from_numpy(data["road"])
        self.assertTrue(torch.allclose(result[:, :, 0], expected, atol=1e-6))

    def test_test_labels_cannot_change_fit_or_checkpoint_choice(self):
        data = self.data()
        index = {"samples": [{"split": s, "workload_id": i} for i, s in enumerate(
            ("train", "validation", "test"))]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, _, _, a = train_one(data, index, Path(directory) / "a", epochs=1,
                                       device="cpu", batch_size=1, threads=1)
            data["target"][2, :, 0, 4, 4] = 100
            data["target"][2, :, 1, 4, 4] = 50
            second, _, _, b = train_one(data, index, Path(directory) / "b", epochs=1,
                                        device="cpu", batch_size=1, threads=1)
        self.assertEqual(a["best_validation_loss"], b["best_validation_loss"])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(
            first.state_dict().values(), second.state_dict().values())))


if __name__ == "__main__":
    unittest.main()
