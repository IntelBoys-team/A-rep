import tempfile
import py7zr
from pathlib import Path

from archive_validator import validate_and_extract, ValidationError

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    archive = root / "bad.7z"
    png = root / "sample.png"
    txt = root / "notes.txt"
    cache = root / "cache"

    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")
    txt.write_text("Это запрещённый файл", encoding="utf-8")

    with py7zr.SevenZipFile(archive, "w") as zf:
        zf.write(png, arcname="sample.png")
        zf.write(txt, arcname="notes.txt")

    try:
        validate_and_extract(archive, cache)
        print("Тест TXT — FAIL: архив не должен пройти")
    except ValidationError as e:
        print("Тест TXT — PASS: архив заблокирован")
        print("Причина:", e)
        print("Кэш очищен:", not cache.exists())
