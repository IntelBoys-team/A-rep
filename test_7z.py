
import tempfile
import py7zr
from pathlib import Path

from archive_validator import validate_and_extract, ValidationError


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    archive = root / "test.7z"
    source = root / "sample.png"
    cache = root / "cache"

    # Для проверки сигнатуры достаточно заголовка PNG
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")

    with py7zr.SevenZipFile(archive, "w") as zf:
        zf.write(source, arcname="sample.png")

    try:
        result = validate_and_extract(archive, cache)
        print("Тест 7z — PNG: PASS")
        print("Найденные файлы:", [p.name for p in result])
    except Exception as e:
        print("Тест 7z — PNG: FAIL")
        print(type(e).__name__, e)
