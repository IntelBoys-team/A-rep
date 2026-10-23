
import tempfile
import zipfile
from pathlib import Path

import pytest

from archive_validator import validate_and_extract, ValidationError


def test_broken_zip_is_rejected_and_cache_cleaned():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cache = root / "cache"
        broken_zip = root / "broken.zip"
        broken_zip.write_bytes(b"this is not a zip archive")

        with pytest.raises(ValidationError):
            validate_and_extract(broken_zip, cache)

        assert not cache.exists()


def test_png_archive_is_accepted():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cache = root / "cache"
        good_zip = root / "good.zip"

        with zipfile.ZipFile(good_zip, "w") as z:
            z.writestr("image.png", b"\x89PNG\r\n\x1a\n" + b"test")

        files = validate_and_extract(good_zip, cache)

        assert files


def test_txt_inside_zip_is_rejected_and_cache_cleaned():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cache = root / "cache"
        bad_zip = root / "bad.zip"

        with zipfile.ZipFile(bad_zip, "w") as z:
            z.writestr("image.png", b"\x89PNG\r\n\x1a\n" + b"test")
            z.writestr("notes.txt", "hello")

        with pytest.raises(ValidationError):
            validate_and_extract(bad_zip, cache)

        assert not cache.exists()


def test_exe_inside_zip_is_rejected():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cache = root / "cache"
        bad_zip = root / "bad_exe.zip"

        with zipfile.ZipFile(bad_zip, "w") as z:
            z.writestr("program.exe", b"MZ" + b"\x00" * 20)

        with pytest.raises(ValidationError):
            validate_and_extract(bad_zip, cache)
