"""Проверяет целостность файлов сразу после распаковки поставленного ZIP.

После собственных правок несовпадение ожидаемо. Хеши защищают от случайной
порчи и неполного копирования, но не заменяют цифровую подпись отправителя.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    manifest = ROOT / "MANIFEST.sha256.json"
    if not manifest.is_file():
        raise SystemExit("Нет MANIFEST.sha256.json: проверяйте именно распакованный архив поставки.")
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    errors = []
    for name, checksum in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append(name + ": отсутствует или неверный путь")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != checksum:
            errors.append(name + ": файл изменён")
    print(json.dumps({"checked": len(expected), "errors": errors,
                      "passed": not errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
