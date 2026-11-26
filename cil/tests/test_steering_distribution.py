from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distribution_model import DistributionCIL, distribution_from_checkpoint
from drive import continuous_steering_mode, local_steering_mode, reached_goal, steering_median, temporal_steering_mode
from ordinal_model import OrdinalCIL
from turn_ordinal_model import TurnOrdinalCIL


class SteeringDistributionTests(unittest.TestCase):
    def test_turn_distribution_keeps_expected_steering_and_mse_on_straight_commands(self):
        torch.set_num_threads(1)
        model = TurnOrdinalCIL().eval()
        commands = torch.arange(4)
        output = model(torch.zeros(4, 3, 88, 200), torch.zeros(4, 1), commands)
        target = torch.tensor([.1, .3, -.1, -.2])
        candidates = output["steering_candidates"]
        mean = (output["steering_logits"].softmax(dim=-1) * candidates).sum(dim=-1)
        expected_modes = candidates.gather(1, output["steering_logits"].argmax(dim=1, keepdim=True))[:, 0]
        self.assertEqual(output["turn_branch"].tolist(), [False, True, False, True])
        mask = output["turn_branch"]
        torch.testing.assert_close(output["control"][:, 0], torch.where(mask, expected_modes, mean))
        loss = model.steering_loss(output, target)
        torch.testing.assert_close(loss[~mask], (10 * (mean - target).square())[~mask])
        torch.testing.assert_close(loss[mask], OrdinalCIL.steering_loss(model, output, target)[mask])

    def test_local_mode_combines_adjacent_peaks_and_handles_a_boundary(self):
        probabilities = torch.tensor([.31, .26, .01, .01, .41])
        candidates = torch.tensor([-.8, -.4, 0., .4, .8])
        torch.testing.assert_close(local_steering_mode(probabilities, candidates), torch.tensor(-.352 / .58))
        probabilities = torch.tensor([0., 0., 0., 0., 1.])
        self.assertEqual(local_steering_mode(probabilities, candidates), candidates[-1])

    def test_stop_at_projected_endpoint_also_requires_positional_radius(self):
        far = {"x_m": 130., "y_m": 0.}
        near = {"x_m": 100.8, "y_m": 0.6}
        self.assertFalse(reached_goal(far, 100., 0., 100., (100., 0.), 1.5, 2.))
        self.assertTrue(reached_goal(near, 100., .6, 100., (100., 0.), 1.5, 2.))

    def test_temporal_mode_accumulates_evidence_but_accepts_a_new_branch(self):
        candidates = torch.tensor([-.45, .45])
        previous = torch.tensor([.95, .05])
        current = torch.tensor([.1, .9])
        angle, filtered = temporal_steering_mode(current, candidates, previous, .4)
        self.assertEqual(angle, candidates[0])
        angle, _ = temporal_steering_mode(current, candidates, filtered, .4)
        self.assertEqual(angle, candidates[1])
        angle, _ = temporal_steering_mode(current, candidates, None, .4)
        self.assertEqual(angle, candidates[1])

    def test_median_uses_mass_across_bins(self):
        probabilities = torch.tensor([.39, .20, .23, .18])
        candidates = torch.tensor([-.5, -.2, .1, .4])
        self.assertEqual(steering_median(probabilities, candidates), candidates[1])

    def test_ordinal_loss_penalizes_probability_farther_from_the_target(self):
        baseline = DistributionCIL("bins")
        ordinal = OrdinalCIL()
        target = torch.zeros(2)
        middle = ordinal.components // 2
        logits = torch.full((2, ordinal.components), -12.0)
        logits[:, middle] = 2.0
        logits[0, middle + 1] = logits[1, middle + 4] = 1.0
        output = {
            "steering_logits": logits,
            "steering_candidates": ordinal.bin_centres.expand(2, -1),
        }

        self.assertAlmostEqual(*baseline.steering_loss(output, target).tolist(), places=6)
        near_loss, far_loss = ordinal.steering_loss(output, target)
        self.assertLess(near_loss, far_loss)

    def test_transition_prior_prefers_a_nearby_mode_at_similar_confidence(self):
        candidates = torch.tensor([.03, .38])
        self.assertEqual(continuous_steering_mode(torch.tensor([0., .03]), candidates, .02, .16), candidates[0])
        self.assertEqual(continuous_steering_mode(torch.tensor([0., 6.]), candidates, .02, .16), candidates[1])

    def test_modes_and_loss_use_the_selected_command_branch(self):
        torch.set_num_threads(1)
        images = torch.zeros(4, 3, 88, 200)
        speeds = torch.zeros(4, 1)
        commands = torch.arange(4)
        targets = torch.tensor([-1., -.3, .4, 1.])
        for distribution in ("mixture", "bins"):
            with self.subTest(distribution=distribution):
                model = DistributionCIL(distribution).eval()
                output = model(images, speeds, commands)
                chosen = output["steering_logits"].argmax(dim=1, keepdim=True)
                steer = output["steering_candidates"].gather(1, chosen)[:, 0]
                torch.testing.assert_close(steer, output["control"][:, 0])
                torch.testing.assert_close(output["control"], output["all_controls"][commands, commands])
                self.assertTrue((steer.abs() <= 1.).all())
                loss = model.steering_loss(output, targets).mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for branch in model.branches:
                    self.assertTrue(torch.isfinite(branch[-1].weight.grad).all())
                loaded = distribution_from_checkpoint({"architecture": model.architecture,
                                                       "steering_distribution": model.spec})
                loaded.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
