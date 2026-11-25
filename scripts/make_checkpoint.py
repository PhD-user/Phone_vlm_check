"""Собирает переносимый архив текущего проекта без виртуального окружения.

Не заменяет хранение исходников и фотографий в системе резервного копирования.
Без --include-data фотографии data/raw намеренно исключены, но кэши, разметка,
эксперименты и отчёты сохраняются. Контрольные суммы пересчитываются в архиве.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-data", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("Выходной архив уже существует. Укажите новое имя.")
    if output.is_relative_to(ROOT):
        raise SystemExit("Сохраните архив за пределами папки проекта.")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            raise SystemExit(f"Символическая ссылка не включается автоматически: {relative}")
        if any(part in (".venv", "__pycache__", ".git", ".pytest_cache") for part in relative.parts):
            continue
        if relative.as_posix().startswith("models/download_cache/") or path.suffix in (".pyc", ".part"):
            continue
        if relative.as_posix() == "MANIFEST.sha256.json":
            continue
        if not args.include_data and relative.as_posix().startswith("data/raw/") and path.name != "README.txt":
            continue
        files.append((path, relative))
    manifest = {relative.as_posix(): digest(path) for path, relative in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in files:
            archive.write(path, (Path(ROOT.name) / relative).as_posix())
        archive.writestr(f"{ROOT.name}/MANIFEST.sha256.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    partial.replace(output)
    print(json.dumps({"archive": str(output), "files": len(files) + 1,
                      "bytes": output.stat().st_size, "sha256": digest(output)}, indent=2))


if __name__ == "__main__":
    main()
