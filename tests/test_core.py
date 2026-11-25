"""Существенные проверки данных и математики без загрузки базовой нейросети.

Запуск: python -m unittest discover -s tests -v
Временные фотографии создаются автоматически и не являются датасетом телефонов.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from phone_vlm.common import ProjectError, load_config, sha256
from phone_vlm.dataset import (assert_no_leakage, audit, groups, read_manifest,
                               split_dataset, write_manifest)
from phone_vlm.features import feature_contract, load_features
from phone_vlm.images import preprocess, read_rgb
from phone_vlm.metrics import classification_metrics, probabilities


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_config()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_portrait_and_source_preservation(self):
        path = self.root / "portrait.jpg"
        Image.new("RGB", (480, 640), "red").save(path)
        before = sha256(path)
        image = read_rgb(path, self.config["image_policy"])
        self.assertEqual(image.size, (480, 640))
        preprocess(image, self.config["preprocess"])
        self.assertEqual(sha256(path), before)

    def test_letterbox_keeps_object_proportions(self):
        image = Image.new("RGB", (640, 480), "white")
        ImageDraw.Draw(image).rectangle((240, 80, 399, 399), fill="red")
        tensor = preprocess(image, self.config["preprocess"])
        self.assertEqual(tensor.shape, (3, 256, 256))
        red = (tensor[0] > .9) & (tensor[1] < .1) & (tensor[2] < .1)
        y, x = np.where(red)
        self.assertAlmostEqual((y.max() - y.min() + 1) / (x.max() - x.min() + 1), 2, delta=.06)

    def test_invalid_files(self):
        cases = [self.root / "missing.jpg", self.root / "broken.jpg", self.root / "tiny.png"]
        cases[1].write_bytes(b"not an image")
        Image.new("RGB", (4, 4)).save(cases[2])
        for path in cases:
            with self.subTest(path=path), self.assertRaises(ProjectError):
                read_rgb(path, self.config["image_policy"])

    def test_exif_orientation(self):
        path = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (640, 480), "red").save(path, exif=exif)
        self.assertEqual(read_rgb(path, self.config["image_policy"]).size, (480, 640))


class DatasetTests(unittest.TestCase):
    def test_transitive_groups_and_cross_split_leak(self):
        rows = [dict(path="a", device_id="one", session_id="s1", sha256="a", label="android", split="train"),
                dict(path="b", device_id="one", session_id="s2", sha256="b", label="android", split="train"),
                dict(path="c", device_id="two", session_id="s2", sha256="c", label="iphone", split="test")]
        self.assertEqual(len(groups(rows)), 1)
        with self.assertRaisesRegex(ProjectError, "Утечка"):
            assert_no_leakage(rows)

    def test_grouped_split_is_reproducible(self):
        rows = []
        for label in ("android", "iphone"):
            for i in range(30):
                for frame in range(2):
                    rows.append(dict(path=f"{label}/{i}_{frame}.jpg", label=label,
                                     device_id=f"{label}_{i}", session_id=f"{label}_{i}",
                                     sha256=f"{label}_{i}_{frame}", split=""))
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            ra = split_dataset(rows, a, trials=100)
            rb = split_dataset(rows, b, trials=100)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(ra, rb)
            parsed = read_manifest(a, Path(tmp), require_split=True)
            assert_no_leakage(parsed)
            self.assertEqual(sum(sum(x.values()) for x in ra["counts"].values()), 120)

    def test_missing_group_ids_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "labels.csv"
            write_manifest(manifest, [{"path": "a.jpg", "label": "android"}])
            with self.assertRaises(ProjectError):
                read_manifest(manifest, root)
            write_manifest(manifest, [{"path": "../outside.jpg", "label": "android", "device_id": "a", "session_id": "s"}])
            with self.assertRaises(ProjectError):
                read_manifest(manifest, root)

    def test_conflicting_duplicate_labels_rejected(self):
        config, _ = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (640, 480), "red").save(root / "a.png")
            (root / "b.png").write_bytes((root / "a.png").read_bytes())
            rows = [{"path": f"{name}.png", "label": label, "device_id": name, "session_id": name}
                    for name, label in [("a", "android"), ("b", "iphone")]]
            with self.assertRaisesRegex(ProjectError, "разными классами"):
                audit(rows, root, config["image_policy"], root / "audit.csv")


class MetricTests(unittest.TestCase):
    def test_warnings_do_not_hide_classification_errors(self):
        report = classification_metrics(np.array([0, 1]), np.array([[.9, .1], [.51, .49]]), .6)
        self.assertEqual(report["accuracy"], .5)
        self.assertEqual(report["confident_coverage"], .5)
        self.assertEqual(report["accuracy_on_confident"], 1)
        self.assertEqual(report["confusion_matrix"], [[1, 0], [1, 0]])

    def test_softmax_is_stable_for_large_logits(self):
        result = probabilities(np.array([[1000000.]]), np.array([[1., -1.]]), np.zeros(2))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, [[1, 0]])


if __name__ == "__main__":
    unittest.main()
