import unittest

import numpy as np
import torch

from rainbow import NoisyLinear, Replay, project


class RainbowTests(unittest.TestCase):
    def test_projection_preserves_mass_including_exact_atoms(self):
        support = torch.linspace(-10, 10, 51)
        probabilities = torch.rand(3, 51)
        probabilities /= probabilities.sum(1, keepdim=True)
        result = project(probabilities, torch.tensor([0., 0., 100.]),
                         torch.tensor([1., 0., 0.]), support)
        torch.testing.assert_close(result.sum(1), torch.ones(3))
        torch.testing.assert_close(result[0], probabilities[0], atol=1e-5, rtol=1e-4)
        self.assertAlmostEqual(result[1, 25].item(), 1., places=5)
        self.assertAlmostEqual(result[2, -1].item(), 1., places=5)

    def test_terminal_flush_and_truncation_bootstrap(self):
        state = np.zeros((4, 84, 84), dtype=np.uint8)
        replay = Replay(8, gamma=0.5)
        replay.add(state, 0, 1, state, False, False)
        replay.add(state, 0, 2, state, True, False)
        self.assertEqual(len(replay.data), 2)
        self.assertEqual(len(replay.pending), 0)
        self.assertEqual(replay.data[0][2], 2)
        self.assertEqual(replay.data[0][4], 0)
        replay.add(state, 0, 3, state, False, True)
        self.assertEqual(replay.data[2][4], 0.5)
        self.assertEqual(replay.data[2][2], 3)

    def test_priorities_and_importance_weights(self):
        replay = Replay(2, n=1)
        state = np.zeros(1, dtype=np.uint8)
        for _ in range(3):
            replay.add(state, 0, 0, state, True, False)
        replay.update(np.array([0, 0, 1]), np.array([1., 3., 2.]))
        self.assertAlmostEqual(replay.priorities[0], 3.000001)
        _, _, weights = replay.sample(8, beta=1)
        self.assertTrue(torch.all((weights > 0) & (weights <= 1)))

    def test_noisy_exploration_disabled_for_evaluation(self):
        layer = NoisyLinear(3, 4)
        x = torch.ones(2, 3)
        first = layer(x)
        layer.reset_noise()
        self.assertFalse(torch.equal(first, layer(x)))
        layer.eval()
        first = layer(x)
        layer.reset_noise()
        torch.testing.assert_close(first, layer(x))


if __name__ == '__main__':
    unittest.main()
