"""Integration tests for target-controlled CAM, using real GPU operations."""
import unittest
import numpy as np
import torch

from experiment import GradCAM, make_splits


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU())
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.layer4(x).mean(dim=(2, 3)))


class ExperimentTests(unittest.TestCase):
    def test_split_disjoint_and_balanced(self):
        labels = np.repeat(np.arange(10), 100)
        splits = make_splits(labels, 100, 42)
        self.assertEqual(len(set(sum(splits.values(), []))), 300)
        for ids in splits.values():
            np.testing.assert_array_equal(np.bincount(labels[ids]), np.full(10, 10))
        self.assertEqual(splits, make_splits(labels, 100, 42))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for actual integration test")
    def test_cam_batch_matches_individual(self):
        torch.manual_seed(13)
        model = SmallModel().cuda().eval()
        extractor = GradCAM(model)
        inputs = torch.rand(3, 3, 12, 12, device="cuda")
        targets = torch.tensor([0, 1, 0], device="cuda")
        logits, own, fixed = extractor(inputs, targets)
        for i in range(3):
            li, oi, fi = extractor(inputs[i:i+1], targets[i:i+1])
            np.testing.assert_allclose(fixed[i], fi[0], atol=2e-5)
            np.testing.assert_allclose(own[i], oi[0], atol=2e-5)
        self.assertTrue(np.isfinite(fixed).all())
        self.assertTrue(((fixed >= 0) & (fixed <= 1)).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        extractor.close()


if __name__ == "__main__":
    unittest.main()
