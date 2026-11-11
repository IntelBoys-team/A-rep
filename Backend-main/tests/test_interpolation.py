"""Smoke tests for Pillow/SciPy image resizing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_core import convert_image, resize_image


class InterpolationTests(unittest.TestCase):
    def test_scipy_methods_return_requested_size(self):
        source = Image.fromarray(
            np.random.default_rng(7).integers(0, 256, (23, 31, 4), dtype=np.uint8),
            mode="RGBA",
        )
        for method in ("Nearest", "Linear", "Cubic", "Quintic"):
            with self.subTest(method=method):
                result = resize_image(source, (17, 13), "SciPy", method)
                self.assertEqual(result.size, (17, 13))
                self.assertEqual(result.mode, "RGBA")

    def test_pillow_lanczos_still_works(self):
        source = Image.new("RGB", (20, 10), "red")
        result = resize_image(source, (10, 5), "Pillow", "Lanczos")
        self.assertEqual(result.size, (10, 5))
        self.assertEqual(result.mode, "RGB")

    def test_scipy_uses_pixel_center_convention(self):
        # Как у Pillow: крайние пиксели сохраняются, промежуточные
        # интерполируются по центрам, а не по краям массива.
        source = Image.fromarray(
            np.array([[0, 100]], dtype=np.uint8),
            mode="L",
        )
        result = resize_image(source, (4, 1), "SciPy", "Linear")
        self.assertEqual(
            np.asarray(result).ravel().tolist(),
            [0, 25, 75, 100],
        )

    def test_convert_image_supports_both_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            Image.new("RGB", (32, 24), "red").save(src)

            for library, method in (("Pillow", "Lanczos"), ("SciPy", "Cubic")):
                with self.subTest(library=library):
                    out = convert_image(
                        src,
                        Path(tmp) / library,
                        output_format="PNG",
                        resize_enabled=True,
                        width=16,
                        height=12,
                        keep_aspect=True,
                        interpolation=method,
                        interpolation_library=library,
                        enhance_enabled=False,
                    )
                    self.assertTrue(out.exists())
                    with Image.open(out) as result:
                        self.assertEqual(result.size, (16, 12))

    def test_incompatible_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            Image.new("RGB", (8, 8), "red").save(src)
            with self.assertRaises(ValueError):
                convert_image(
                    src,
                    Path(tmp) / "out",
                    interpolation_library="SciPy",
                    interpolation="Lanczos",
                )


if __name__ == "__main__":
    unittest.main()
