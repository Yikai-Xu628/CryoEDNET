"""Small synthetic software tests, NOT experiments or pretrained checkpoints."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cryoednet import CryoEDNet
from cryoednet.model import MultiScale, EdgeEnhancement, DetailPreservation
from cryoednet.perceptual import FeatureExtractor, Distiller, load_extractor
from cryoednet.losses import CombinedLoss, SSIMLoss
from cryoednet.data import Stack, stack_stats, normalize, denormalize, PairDataset
from cryoednet.utils import sha256
import train
import infer
import evaluate

torch.set_num_threads(1)


class FakeTeacher(nn.Module):
    """Only for testing graph connectivity without downloading VGG weights."""
    def forward(self, x):
        return [x.repeat(1, 64, 1, 1),
                nn.functional.avg_pool2d(x, 2).repeat(1, 128, 1, 1),
                nn.functional.avg_pool2d(x, 4).repeat(1, 256, 1, 1)]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.tmp = tempfile.TemporaryDirectory(prefix="cryoednet-test-")
        self.root = Path(self.tmp.name)
        self.clean_path = self.root / "clean.npy"
        self.noisy_path = self.root / "noisy.npy"
        rng = np.random.default_rng(5)
        self.clean = rng.normal(2, 3, (12, 16, 16)).astype(np.float32)
        np.save(self.clean_path, self.clean)
        np.save(self.noisy_path, self.clean+rng.normal(0, 1, self.clean.shape).astype(np.float32))

    def tearDown(self):
        self.tmp.cleanup()

    def stats(self):
        s = Stack(self.clean_path)
        stats = stack_stats(s, batch_size=3)
        s.close()
        return stats

    def test_paper_dimensions(self):
        ms = MultiScale()
        self.assertEqual(ms.branch3.out_channels, 64)
        self.assertEqual(ms.fusion.in_channels, 112)
        self.assertEqual(ms.fusion.kernel_size, (3, 3))
        self.assertEqual(EdgeEnhancement().features[0].out_channels, 32)
        self.assertEqual(DetailPreservation().weight[2].kernel_size, (3, 3))

    def test_residual_and_arbitrary_shape(self):
        model = CryoEDNet()
        nn.init.zeros_(model.output_conv.weight)
        nn.init.zeros_(model.output_conv.bias)
        for h, w in [(16, 16), (17, 19)]:
            x = torch.randn(2, 1, h, w)
            torch.testing.assert_close(model(x), x)

    def test_ablations(self):
        for key in ("use_multiscale", "use_edge", "use_detail"):
            model = CryoEDNet(**{key: False})
            self.assertEqual(model(torch.randn(1, 1, 16, 16)).shape, (1, 1, 16, 16))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            CryoEDNet()(torch.zeros(1, 3, 16, 16))

    def test_stats_and_standardization(self):
        stats = self.stats()
        self.assertAlmostEqual(stats["mean"], float(self.clean.astype(np.float64).mean()), places=10)
        self.assertAlmostEqual(stats["std"], float(self.clean.astype(np.float64).std()), places=10)
        dataset = PairDataset(self.clean_path, self.noisy_path, stats)
        a, b = dataset[0]
        torch.testing.assert_close(a, torch.from_numpy((self.clean[0]-stats["mean"])/stats["std"])[None])
        self.assertEqual(b.shape, a.shape)
        dataset.clean.close()
        dataset.noisy.close()

    def test_normalization_roundtrip(self):
        x = torch.randn(2, 1, 16, 16)
        train_stats = dict(mean=-.05, std=4.34)
        input_stats = dict(mean=2., std=3.)
        for mode in ("train", "stack", "legacy-stack-rescale"):
            torch.testing.assert_close(denormalize(normalize(x, mode, train_stats, input_stats),
                                                   mode, train_stats, input_stats), x)

    def test_constant_and_nonfinite_inputs_rejected(self):
        for value in (0., float("nan")):
            path = self.root / "bad.npy"
            np.save(path, np.full((2, 16, 16), value, dtype=np.float32))
            s = Stack(path)
            with self.assertRaises(ValueError):
                stack_stats(s)
            s.close()

    def test_four_dimensional_input(self):
        path = self.root / "channel.npy"
        np.save(path, self.clean[:, None])
        s = Stack(path)
        self.assertEqual(s.data.shape, self.clean.shape)
        s.close()

    def test_loss_gradients_and_freezing(self):
        extractor = FeatureExtractor()
        loss_fn = CombinedLoss(extractor).train()
        self.assertFalse(extractor.training)
        a = torch.randn(2, 1, 16, 16, requires_grad=True)
        b = torch.randn_like(a)
        loss_fn(a, b).backward()
        self.assertGreater(float(a.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in extractor.parameters()))

    def test_loss_identity_and_coefficients(self):
        loss_fn = CombinedLoss(FeatureExtractor())
        a, b = torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16)
        self.assertLess(abs(float(loss_fn(a, a))), 1e-5)
        c = loss_fn.components(a, b)
        torch.testing.assert_close(loss_fn(a, b), c["rec"]+.1*c["perc"]+.5*c["ssim"]+.1*c["freq"])

    def test_distillation_trains_entire_student(self):
        student = FeatureExtractor()
        distiller = Distiller(student, FakeTeacher())
        distiller(torch.randn(2, 1, 16, 16), torch.rand(2, 1, 16, 16)).backward()
        for name, p in student.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(float(student.cbam.spatial.weight.grad.abs().sum()), 0)
        self.assertGreater(float(student.residual.weight.grad.abs().sum()), 0)

    def test_random_or_legacy_extractor_rejected(self):
        path = self.root / "bad.pt"
        torch.save({"state_dict": FeatureExtractor().state_dict()}, path)
        with self.assertRaises(ValueError):
            load_extractor(path)

    def fixture_extractor(self):
        # This mock exists only in a temporary unit test. It is never exposed
        # as a trained VGG extractor, and load_extractor is patched explicitly.
        return FeatureExtractor(), dict(stats=self.stats(), clean_sha256=sha256(self.clean_path))

    def train_smoke(self, engine):
        out = self.root / engine
        dummy_file = self.root / "mock-extractor-not-pretrained.txt"
        dummy_file.touch()
        with patch.object(train, "load_extractor", side_effect=lambda _: self.fixture_extractor()):
            train.main(["--clean", str(self.clean_path), "--noisy", str(self.noisy_path),
                        "--extractor", str(dummy_file), "--out", str(out), "--engine", engine,
                        "--epochs", "1", "--batch-size", "2", "--device", "cpu"])
        ckpt = torch.load(out / "checkpoint.pt", weights_only=True)
        self.assertEqual(ckpt["scheduler"]["last_epoch"], 6)
        return out

    def test_torch_training_and_npy_inference(self):
        out = self.train_smoke("torch")
        pred = self.root / "prediction.npy"
        infer.main(["--input", str(self.noisy_path), "--checkpoint", str(out/"checkpoint.pt"),
                    "--output", str(pred), "--normalization", "train", "--device", "cpu"])
        result = np.load(pred)
        self.assertEqual(result.shape, self.clean.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(pred.with_suffix(".npy.json").exists())
        evaluate.main(["--clean", str(self.clean_path), "--prediction", str(pred),
                       "--peak", "20", "--output", str(self.root/"metrics.csv")])

    def test_deepinv_training(self):
        self.train_smoke("deepinv")

    def test_mrcs_metadata_and_axis_order(self):
        import mrcfile
        checkpoint = self.root / "identity.pt"
        model = CryoEDNet()
        nn.init.zeros_(model.output_conv.weight)
        nn.init.zeros_(model.output_conv.bias)
        torch.save(dict(format="cryoednet-reconstruction-v1", state_dict=model.state_dict(),
                        architecture=model.config, train_stats=self.stats(),
                        config=dict(provenance="unit-test-only")), checkpoint)
        for mode in ("train", "stack", "legacy-stack-rescale"):
            output = self.root / (mode+".mrcs")
            infer.main(["--input", str(self.clean_path), "--checkpoint", str(checkpoint),
                        "--output", str(output), "--normalization", mode,
                        "--pixel-size", "1.5", "--limit", "3", "--device", "cpu"])
            with mrcfile.open(output) as m:
                np.testing.assert_allclose(m.data, self.clean[:3], atol=2e-6)
                self.assertAlmostEqual(float(m.voxel_size.x), 1.5)
                self.assertTrue(m.is_image_stack())
            s = Stack(output)
            self.assertEqual(s.data.shape, (3, 16, 16))
            s.close()

    def test_inference_does_not_overwrite(self):
        output = self.root / "already.npy"
        output.touch()
        with self.assertRaises(FileExistsError):
            infer.main(["--input", str(self.clean_path), "--checkpoint", "missing.pt",
                        "--output", str(output), "--normalization", "train"])


if __name__ == "__main__":
    unittest.main()
