"""명령별 균형 추출·손실 집계·branch 선택을 작은 독립 예제로 검사한다."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import CILModel
from train import CommandBalancedBatchSampler, configure_dropout, run_epoch


class UnevenDataset:
    def __init__(self, counts=(31, 3, 7, 2)):
        self.samples = [
            (None, None, None, command, None)
            for command, count in enumerate(counts) for _ in range(count)
        ]

    def __len__(self):
        return len(self.samples)


class ZeroModel(torch.nn.Module):
    def forward(self, image, speed, command):
        return {"control": torch.zeros_like(image), "pred_speed": torch.zeros_like(speed)}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_imbalanced_data_has_equal_commands_in_every_batch(self):
        dataset = UnevenDataset()
        sampler = CommandBalancedBatchSampler(dataset, 12, 7)
        batches = list(sampler)
        self.assertEqual(len(batches), math.ceil(43 / 12))
        for batch in batches:
            commands = [dataset.samples[index][3] for index in batch]
            self.assertEqual([commands.count(i) for i in range(4)], [3, 3, 3, 3])

    def test_sampling_is_repeatable_and_changes_with_epoch(self):
        sampler = CommandBalancedBatchSampler(UnevenDataset(), 12, 7)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.epoch = 1
        self.assertNotEqual(first, list(sampler))

    def test_rejects_missing_commands_or_unbalanced_batch_size(self):
        with self.assertRaises(ValueError):
            CommandBalancedBatchSampler(UnevenDataset(), 10, 7)
        with self.assertRaises(ValueError):
            CommandBalancedBatchSampler(UnevenDataset((3, 2, 1, 0)), 12, 7)

    def test_metrics_have_correct_sample_and_command_averages(self):
        rows = [
            {"image": torch.zeros(3), "speed": torch.tensor([.25 + .05 * j]),
             "command": torch.tensor(command),
             "target": torch.tensor([.1 * command, .2 + .01 * j, 0.])}
            for command, count in enumerate((5, 3, 2, 1)) for j in range(count)
        ]
        model = ZeroModel()
        result = run_epoch(model, DataLoader(rows, batch_size=4), torch.device("cpu"), .05)
        targets = np.stack([row["target"].numpy() for row in rows])
        speeds = np.array([row["speed"].item() for row in rows])
        commands = np.array([row["command"].item() for row in rows])
        sample_mse = (targets ** 2).mean(axis=1)
        self.assertAlmostEqual(result["control_mse"], sample_mse.mean(), places=7)
        self.assertAlmostEqual(result["loss"], sample_mse.mean() + .05 * (speeds ** 2).mean(), places=7)
        macro = np.mean([sample_mse[commands == i].mean() for i in range(4)])
        self.assertAlmostEqual(result["mean_command_mse"], macro, places=7)
        self.assertAlmostEqual(result["speed_mae_kph"], speeds.mean() * 20, places=6)
        self.assertFalse(model.training)

    def test_control_gradient_only_updates_the_selected_branch(self):
        torch.manual_seed(7)
        model = CILModel().eval()
        output = model(torch.rand(2, 3, 88, 200), torch.ones(2, 1), torch.tensor([1, 1]))
        output["control"].square().mean().backward()
        for index, branch in enumerate(model.branches):
            magnitude = sum(
                parameter.grad.abs().sum().item()
                for parameter in branch.parameters() if parameter.grad is not None
            )
            if index == 1:
                self.assertGreater(magnitude, 0)
            else:
                self.assertEqual(magnitude, 0)
        self.assertTrue(all(p.grad is None for p in model.speed_head.parameters()))

    def test_dropout_tuning_preserves_layers_weights_and_eval_output(self):
        torch.manual_seed(7)
        model = CILModel().eval()
        modules_before = {name: type(module) for name, module in model.named_modules()}
        shapes_before = {name: value.shape for name, value in model.state_dict().items()}
        inputs = (torch.rand(4, 3, 88, 200), torch.ones(4, 1), torch.arange(4))
        with torch.no_grad():
            before = model(*inputs)["control"].clone()
        counts = configure_dropout(model, .25, .60)
        self.assertEqual(counts, {"conv": 8, "fc": 15})
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                self.assertEqual(module.p, .25 if name.startswith("image_conv.") else .60)
        self.assertEqual(modules_before, {name: type(module) for name, module in model.named_modules()})
        self.assertEqual(shapes_before, {name: value.shape for name, value in model.state_dict().items()})
        with torch.no_grad():
            self.assertTrue(torch.equal(before, model(*inputs)["control"]))

    def test_dropout_tuning_rejects_invalid_probabilities(self):
        model = CILModel()
        for probability in (-.1, 1., float("nan"), float("inf")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                configure_dropout(model, probability, .5)


if __name__ == "__main__":
    unittest.main()
