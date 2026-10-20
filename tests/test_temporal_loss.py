import unittest

import torch

from dinov3.finetune.temporal.loss import masks, sigmoid_loss


class Tests(unittest.TestCase):
    def test_window_is_anchor_relative_not_transitive(self):
        a = torch.tensor([1, 1, 1, 2])
        d = torch.tensor([0, 300, 600, 0])
        ids = torch.arange(4)
        p, n = masks(a, d, ids, ids)
        self.assertTrue(p[0, 1])
        self.assertTrue(p[1, 2])
        self.assertFalse(p[0, 2])
        self.assertFalse(n[0, 2])
        self.assertTrue(n[0, 3])
        self.assertFalse(p.diag().any())

    def test_boundary_and_duplicate_family(self):
        a = torch.tensor([1, 1, 1, 2])
        d = torch.tensor([0, 365, 366, 0])
        p, n = masks(a, d, torch.arange(4), torch.tensor([0, 1, 2, 0]))
        self.assertTrue(p[0, 1])
        self.assertFalse(p[0, 2])
        self.assertFalse(n[0, 3])

    def test_finite_gradients_and_scale_bias(self):
        z = torch.randn(8, 16, requires_grad=True)
        t = torch.tensor(10.0, requires_grad=True)
        b = torch.tensor(-5.0, requires_grad=True)
        p, n = masks(torch.arange(8) // 2, torch.zeros(8), torch.arange(8), torch.arange(8))
        loss = sigmoid_loss(z, t, b, p, n)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(torch.isfinite(t.grad))
        self.assertTrue(torch.isfinite(b.grad))


if __name__ == "__main__":
    unittest.main()
