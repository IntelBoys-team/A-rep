
"""PixelForge — обработка изображений: масштабирование, улучшение, конвертация."""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


_RESAMPLING = getattr(Image, "Resampling", Image)

INTERPOLATION_MAP = {
    "Bilinear": _RESAMPLING.BILINEAR,
    "Bicubic": _RESAMPLING.BICUBIC,
    "Lanczos": _RESAMPLING.LANCZOS,
}


SCIPY_INTERPOLATION_MAP = {
    "Nearest": 0,
    "Linear": 1,
    "Cubic": 3,
    "Quintic": 5,
}


def _resize_with_scipy(
    image: Image.Image,
    size: tuple[int, int],
    interpolation: str,
) -> Image.Image:
    """Масштабирует изображение SciPy ndimage с защитой альфа-канала.

    Для уменьшения предварительно применяет Gaussian anti-aliasing.
    Pillow остаётся ответственным за декодирование и сохранение файлов.
    """
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "Для интерполяции SciPy установите зависимости: "
            "pip install numpy scipy"
        ) from exc

    if interpolation not in SCIPY_INTERPOLATION_MAP:
        raise ValueError(f"Метод {interpolation!r} недоступен для SciPy")

    target_w, target_h = size
    source_w, source_h = image.size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Размеры изображения должны быть положительными")

    # SciPy работает с числовыми массивами. Приводим редкие режимы Pillow
    # к стандартному RGB/RGBA, сохраняя прозрачность там, где она есть.
    has_alpha = _has_alpha(image)
    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGBA" if has_alpha else "RGB")

    arr = np.asarray(image).astype(np.float32)
    alpha_channel = arr.ndim == 3 and arr.shape[2] == 4

    # Для RGBA интерполируем premultiplied RGB, чтобы на границах
    # прозрачных пикселей не появлялись светлые/тёмные ореолы.
    if alpha_channel:
        alpha = arr[..., 3:4] / 255.0
        arr[..., :3] *= alpha

    zoom_y = target_h / source_h
    zoom_x = target_w / source_w

    # Антиалиасинг нужен при уменьшении, иначе высокие частоты могут
    # превращаться в муар и ступенчатые артефакты.
    sigma_y = max(0.0, (1.0 / zoom_y - 1.0) * 0.5) if zoom_y < 1 else 0.0
    sigma_x = max(0.0, (1.0 / zoom_x - 1.0) * 0.5) if zoom_x < 1 else 0.0
    if sigma_x or sigma_y:
        sigma = (sigma_y, sigma_x, 0.0) if arr.ndim == 3 else (sigma_y, sigma_x)
        arr = ndimage.gaussian_filter(arr, sigma=sigma, mode="reflect")

    order = SCIPY_INTERPOLATION_MAP[interpolation]
    factors = (zoom_y, zoom_x, 1.0) if arr.ndim == 3 else (zoom_y, zoom_x)
    # grid_mode=True использует центры пикселей (как Pillow): без него
    # ndimage.zoom привязывается к краям и при уменьшении «выбрасывает»
    # половину пикселей, а при увеличении сдвигает изображение.
    result = ndimage.zoom(
        arr,
        factors,
        order=order,
        mode="reflect",
        prefilter=order > 1,
        grid_mode=True,
    )

    # Подгоняем результат точно под запрошенный размер: zoom округляет
    # выходную форму в зависимости от входных размеров и коэффициента.
    result = result[:target_h, :target_w, ...]
    pad_h = target_h - result.shape[0]
    pad_w = target_w - result.shape[1]
    if pad_h > 0 or pad_w > 0:
        pad_width = ((0, max(0, pad_h)), (0, max(0, pad_w)))
        if result.ndim == 3:
            pad_width += ((0, 0),)
        result = np.pad(result, pad_width, mode="edge")

    if alpha_channel:
        alpha = np.clip(result[..., 3:4], 0, 255)
        rgb = np.zeros_like(result[..., :3])
        np.divide(
            result[..., :3] * 255.0,
            alpha,
            out=rgb,
            where=alpha > 1e-6,
        )
        result = np.concatenate((rgb, alpha), axis=2)

    result = np.clip(result, 0, 255).round().astype(np.uint8)
    output_mode = "L" if result.ndim == 2 else ("RGBA" if alpha_channel else "RGB")
    resized = Image.fromarray(result, mode=output_mode)
    resized.info.update(image.info)
    return resized


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    library: str = "Pillow",
    interpolation: str = "Lanczos",
) -> Image.Image:
    """Единая точка масштабирования для Pillow и SciPy."""
    if library == "Pillow":
        if interpolation not in INTERPOLATION_MAP:
            raise ValueError(f"Метод {interpolation!r} недоступен для Pillow")
        return image.resize(size, resample=INTERPOLATION_MAP[interpolation])
    if library == "SciPy":
        return _resize_with_scipy(image, size, interpolation)
    raise ValueError(f"Неизвестная библиотека интерполяции: {library}")

OUTPUT_EXT = {
    "WebP": ".webp",
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "ICO": ".ico",
    "GIF": ".gif",
    "TIFF": ".tiff",
    "AVIF": ".avif",
}

_PIL_FORMAT = {
    name: {"WebP": "WEBP", "JPEG": "JPEG"}.get(name, name)
    for name in OUTPUT_EXT
}

_QUALITY_FORMATS = {"WebP", "JPEG", "AVIF"}

_SAFE_MODES = {
    "WEBP": ("RGB", "RGBA"),
    "PNG": ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"),
    "BMP": ("1", "L", "P", "RGB"),
    "ICO": ("RGB", "RGBA"),
    "GIF": ("L", "P", "RGB", "RGBA"),
    "TIFF": ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "I", "I;16"),
    "AVIF": ("L", "RGB", "RGBA"),
}

_DEST_LOCK = threading.Lock()
_RESERVED_DESTINATIONS: set[Path] = set()


def available_output_formats() -> list[str]:
    """Возвращает форматы, поддерживаемые текущей сборкой Pillow."""
    Image.init()
    return [
        name
        for name, encoder in _PIL_FORMAT.items()
        if encoder in Image.SAVE
    ]


def compute_target_size(
    orig_w: int,
    orig_h: int,
    req_w: int | None,
    req_h: int | None,
    keep_aspect: bool,
) -> tuple[int, int]:
    """Вычисляет разрешение без обрезки изображения."""
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Исходные размеры должны быть положительными")

    if req_w is not None and req_w <= 0:
        raise ValueError("Ширина должна быть положительной")

    if req_h is not None and req_h <= 0:
        raise ValueError("Высота должна быть положительной")

    if req_w is None and req_h is None:
        return orig_w, orig_h

    if keep_aspect:
        if req_w is not None and req_h is not None:
            factor = min(req_w / orig_w, req_h / orig_h)
            return (
                max(1, round(orig_w * factor)),
                max(1, round(orig_h * factor)),
            )

        if req_w is not None:
            return req_w, max(1, round(orig_h * req_w / orig_w))

        assert req_h is not None
        return max(1, round(orig_w * req_h / orig_h)), req_h

    return (
        req_w if req_w is not None else orig_w,
        req_h if req_h is not None else orig_h,
    )


def _has_alpha(image: Image.Image) -> bool:
    return (
        "A" in image.getbands()
        or "transparency" in image.info
    )


def _fit_mode(image: Image.Image, encoder: str) -> Image.Image:
    """Преобразует изображение в режим, поддерживаемый кодеком."""
    permitted = _SAFE_MODES.get(encoder)

    if not permitted or image.mode in permitted:
        return image

    target = (
        "RGBA"
        if _has_alpha(image) and "RGBA" in permitted
        else "RGB"
    )

    if target not in permitted:
        target = "RGB" if "RGB" in permitted else permitted[0]

    return image.convert(target)


def _enhance_image(
    image: Image.Image,
    strength: float,
) -> Image.Image:
    """Улучшает контраст, насыщенность и резкость.

    Альфа-канал сохраняется. 16-битные изображения не переводятся
    в 8-битные ради фильтра.
    """
    if not 0 <= strength <= 2:
        raise ValueError("Сила улучшения должна быть от 0 до 2")

    if strength == 0 or image.mode in ("I;16", "I", "F"):
        return image

    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert(
            "RGBA" if _has_alpha(image) else "RGB"
        )

    alpha = (
        image.getchannel("A")
        if "A" in image.getbands()
        else None
    )

    if image.mode == "RGBA":
        color = image.convert("RGB")
    elif image.mode == "LA":
        color = image.getchannel("L")
    else:
        color = image

    # Умеренная коррекция контраста.
    color = ImageEnhance.Contrast(color).enhance(
        1 + 0.055 * strength
    )

    # Небольшое усиление насыщенности.
    if color.mode == "RGB":
        color = ImageEnhance.Color(color).enhance(
            1 + 0.035 * strength
        )

    # Нерезкая маска: подчёркивает границы и текстуры.
    color = color.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
            percent=round(115 * strength),
            threshold=3,
        )
    )

    if alpha is not None:
        color.putalpha(alpha)

    return color


def _flatten_for_jpeg(image: Image.Image) -> Image.Image:
    """Заменяет прозрачность белым фоном при сохранении JPEG."""
    if _has_alpha(image):
        rgba = image.convert("RGBA")
        bg = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255),
        )
        return Image.alpha_composite(bg, rgba).convert("RGB")

    return image if image.mode == "RGB" else image.convert("RGB")


def _reserve_destination(
    out_dir: Path,
    stem: str,
    extension: str,
) -> Path:
    """Защищает от совпадения имён при параллельной конвертации."""
    with _DEST_LOCK:
        index = 0

        while True:
            suffix = f"_{index}" if index else ""
            candidate = out_dir / f"{stem}{suffix}{extension}"

            if (
                not candidate.exists()
                and candidate not in _RESERVED_DESTINATIONS
            ):
                _RESERVED_DESTINATIONS.add(candidate)
                return candidate

            index += 1


def convert_image(
    src_path: str | Path,
    dst_dir: str | Path,
    output_format: str = "WebP",
    quality: int = 95,
    resize_enabled: bool = False,
    width: int | None = None,
    height: int | None = None,
    keep_aspect: bool = True,
    interpolation: str = "Lanczos",
    interpolation_library: str = "Pillow",
    enhance_enabled: bool = True,
    enhance_strength: float = 1.0,
    lossless_webp: bool = False,
) -> Path:
    """Конвертирует и улучшает изображение.

    Улучшение выполняется даже при resize_enabled=False.
    Обрабатывается первый кадр анимации или TIFF-документа.
    """
    src = Path(src_path)
    out_dir = Path(dst_dir)

    fmt = (
        output_format
        if output_format in OUTPUT_EXT
        else "WebP"
    )

    encoder = _PIL_FORMAT[fmt]

    if fmt not in available_output_formats():
        extra = (
            " (проверьте поддержку AVIF в Pillow)"
            if fmt == "AVIF"
            else ""
        )
        raise ValueError(
            f"Pillow не поддерживает сохранение {fmt}{extra}"
        )

    if interpolation_library == "Pillow":
        if interpolation not in INTERPOLATION_MAP:
            raise ValueError(f"Метод {interpolation!r} недоступен для Pillow")
    elif interpolation_library == "SciPy":
        if interpolation not in SCIPY_INTERPOLATION_MAP:
            raise ValueError(f"Метод {interpolation!r} недоступен для SciPy")
    else:
        raise ValueError(f"Неизвестная библиотека интерполяции: {interpolation_library}")

    quality = max(10, min(100, int(quality)))
    enhance_strength = float(enhance_strength)

    if not 0 <= enhance_strength <= 2:
        raise ValueError(
            "Сила улучшения должна быть от 0 до 2"
        )

    with Image.open(src) as opened:
        opened.seek(0)
        opened.load()

        # Учитываем поворот фотографии из EXIF.
        image = ImageOps.exif_transpose(opened)

        icc_profile = image.info.get("icc_profile")

        # Масштабирование.
        if resize_enabled:
            target = compute_target_size(
                *image.size,
                width,
                height,
                keep_aspect,
            )

            if target != image.size:
                image = resize_image(
                    image,
                    target,
                    library=interpolation_library,
                    interpolation=interpolation,
                )

        # Реальное визуальное улучшение изображения.
        if enhance_enabled:
            image = _enhance_image(
                image,
                enhance_strength,
            )

        kwargs: dict = {}

        if fmt in _QUALITY_FORMATS:
            kwargs["quality"] = quality

        # По возможности сохраняем цветовой профиль.
        if icc_profile and fmt in (
            "WebP", "JPEG", "PNG", "TIFF", "AVIF"
        ):
            kwargs["icc_profile"] = icc_profile

        if fmt == "JPEG":
            image = _flatten_for_jpeg(image)
            kwargs.update(
                optimize=True,
                subsampling=0,
            )

        elif fmt == "WebP":
            image = _fit_mode(image, "WEBP")
            kwargs.update(
                method=6,
                lossless=bool(lossless_webp),
            )

        elif fmt == "PNG":
            image = _fit_mode(image, "PNG")
            kwargs["optimize"] = True

        elif fmt == "ICO":
            image = _fit_mode(image, "ICO")

            if max(image.size) > 256:
                size = compute_target_size(
                    *image.size,
                    256,
                    256,
                    True,
                )
                image = image.resize(
                    size,
                    _RESAMPLING.LANCZOS,
                )

            kwargs["sizes"] = [image.size]

        elif fmt == "BMP":
            image = _flatten_for_jpeg(image)

        else:
            image = _fit_mode(image, encoder)

            if fmt == "TIFF":
                kwargs["compression"] = "tiff_adobe_deflate"

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = _reserve_destination(
            out_dir,
            src.stem,
            OUTPUT_EXT[fmt],
        )

        scratch: Path | None = None

        try:
            # Временный файл, затем атомарное перемещение.
            fd, name = tempfile.mkstemp(
                prefix=".pixelforge-",
                suffix=OUTPUT_EXT[fmt],
                dir=out_dir,
            )

            scratch = Path(name)
            os.close(fd)

            image.save(
                scratch,
                encoder,
                **kwargs,
            )

            os.replace(
                scratch,
                destination,
            )

            scratch = None
            return destination

        finally:
            if scratch is not None:
                scratch.unlink(missing_ok=True)

            with _DEST_LOCK:
                _RESERVED_DESTINATIONS.discard(
                    destination
                )