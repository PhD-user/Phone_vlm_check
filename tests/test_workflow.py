"""Интеграционная проверка полного цикла на искусственных изображениях.

Требует готовый models/base/visual.safetensors, включённый в поставку. Не
обращается к сети. При отсутствии весов тест явно пропускается. Качество
на этих картинках не имеет отношения к точности на реальных смартфонах.
"""

import contextlib
import copy
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from phone_vlm.common import CLASSES, ProjectError, fingerprint, load_config, sha256
from phone_vlm.dataset import audit, read_manifest, split_dataset, write_manifest
from phone_vlm.evaluation import benchmark, evaluate
from phone_vlm.features import extract_features, feature_contract, load_features
from phone_vlm.inference import Recognizer, predict_folder
from phone_vlm.training import train_head


class WorkflowTests(unittest.TestCase):
    @unittest.skipUnless((load_config()[1] / "models/base/visual.safetensors").is_file(), "Нет базовых весов")
    def test_real_backbone_pipeline_offline(self):
        config, source_root = load_config()
        config = copy.deepcopy(config)
        config["base_dir"] = str(source_root / "models/base")
        config["batch_size"] = 8
        with tempfile.TemporaryDirectory() as tmp, patch.object(socket.socket, "connect", side_effect=AssertionError("Сеть запрещена в тесте")):
            root = Path(tmp)
            rows = []
            for label_index, label in enumerate(CLASSES):
                for i in range(8):
                    path = root / f"{label}_{i}.png"
                    image = Image.new("RGB", (640, 480), (25 + i * 8, 70, 150))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((170, 60, 450, 420), fill=(200, 20 + i * 10, 20) if label_index else (20, 190, 20 + i * 10))
                    draw.ellipse((190 + i, 80, 230 + i, 120), fill="black")
                    image.save(path)
                    rows.append(dict(path=path.name, label=label, device_id=f"{label}_{i}", session_id=f"{label}_{i}", split=""))
            with contextlib.redirect_stdout(io.StringIO()):
                audit(rows, root, config["image_policy"], root / "audit.csv")
                audited = read_manifest(root / "audit.csv", root)
                split_dataset(audited, root / "data/splits.csv", trials=100)
                extract_features(config, root, root / config["features"], "cpu")
                trained = train_head(config, root, root / "runs/a")
                report = evaluate(config, root, root / "runs/a", root / "reports/test", "test")
            self.assertGreater(report["n"], 0)
            self.assertFalse(report["acceptance"]["classification_passed"])  # Мало данных для ТЗ.
            recognizer = Recognizer(root / "runs/a", config, root, "cpu")
            result = recognizer.predict(root / rows[0]["path"])
            self.assertIn(result["class"], CLASSES)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1)
            self.assertEqual(recognizer.predict(root / "missing.jpg")["status"], "error")
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "valid.png").write_bytes((root / rows[0]["path"]).read_bytes())
            (inputs / "broken.txt").write_text("not an image")
            batch = predict_folder(recognizer, inputs, root / "reports/batch.csv", root / "reports/log.jsonl")
            self.assertEqual(batch["files"], 2)
            self.assertEqual(batch["errors"], 1)
            self.assertEqual(len((root / "reports/log.jsonl").read_text().splitlines()), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                speed = benchmark(config, root, root / "runs/a", root / "reports/speed", "cpu", 100)
            self.assertEqual(speed["technical_success_rate"], 1)
            self.assertFalse(speed["acceptance_100_images"])
            self.assertGreater(speed["rss_peak_bytes_sampled"], 0)

            # Существенная проверка: изменение ТОЛЬКО тестовых меток не должно
            # менять веса и выбранный C. Кэш перестраиваем по новому контракту.
            x, y, splits, _ = load_features(root / config["features"], config, root)
            new_rows = read_manifest(root / config["manifest"], root)
            for row in new_rows:
                if row["split"] == "test":
                    row["label"] = CLASSES[1 - CLASSES.index(row["label"])]
            write_manifest(root / config["manifest"], new_rows)
            y[splits == "test"] = 1 - y[splits == "test"]
            contract = feature_contract(new_rows, config, sha256(Path(config["base_dir"]) / "visual.safetensors"))
            meta = {"contract": contract, "fingerprint": fingerprint(contract)}
            with (root / config["features"]).open("wb") as stream:
                np.savez_compressed(stream, x=x, y=y, splits=splits, metadata=np.array(json.dumps(meta)))
            with contextlib.redirect_stdout(io.StringIO()):
                second = train_head(config, root, root / "runs/b")
            self.assertEqual(trained["selected_C"], second["selected_C"])
            with np.load(root / "runs/a/head.npz") as a, np.load(root / "runs/b/head.npz") as b:
                np.testing.assert_array_equal(a["weights"], b["weights"])
                np.testing.assert_array_equal(a["bias"], b["bias"])
            # После смены разметки старый эксперимент не оценится на новой версии.
            with self.assertRaises(ProjectError):
                evaluate(config, root, root / "runs/a", root / "reports/wrong", "test")
            # Даже изменение одного байта входа делает кэш недействительным.
            with (root / rows[0]["path"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ProjectError):
                load_features(root / config["features"], config, root)

    def test_cli_reports_missing_training_artifact(self):
        # На старте отсутствующая телефонная голова должна давать понятную
        # диагностику, а не запускать обучение или молча скачивать другую модель.
        _, root = load_config()
        result = subprocess.run([sys.executable, "-m", "phone_vlm", "predict", "absent.jpg",
                                 "--run", "runs/definitely_absent_run_for_test"], cwd=root,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        data = json.loads(result.stderr)
        self.assertEqual(data["status"], "error")
        self.assertIn("run.json", data["message"])


if __name__ == "__main__":
    unittest.main()
