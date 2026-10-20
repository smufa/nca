"""Target pattern loading and preprocessing.

A target is an ``RGBA`` float32 array in ``[0, 1]`` with shape ``[H, W, 4]``
and **premultiplied** RGB (``rgb *= alpha``), matching the reference
implementation. The state grid is padded around the target so the pattern has
room to grow; the padded region has alpha 0.
"""

from __future__ import annotations

import math
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:  # NixOS and some distros do not expose a CA bundle to Python by default
    import certifi

    _SSL_CONTEXT: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CONTEXT = None

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "targets"

_TWEMOJI_URL = (
    "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{code}.png"
)
_NOTO_URLS = (
    "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/3D/png/128/emoji_u{code}.png",
    "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u{code}.png",
)


def _emoji_code(emoji: str) -> str:
    """Convert an emoji string to its twemoji filename code."""
    codes = []
    for ch in emoji:
        cp = ord(ch)
        if cp == 0xFE0F:  # variation selector, not part of the filename
            continue
        codes.append(format(cp, "x"))
    if not codes:
        raise ValueError(f"could not parse emoji {emoji!r}")
    return "-".join(codes)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "nca/0.1"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CONTEXT) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _load_rgba(path: str | Path, max_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]  # premultiply RGB by alpha
    return arr


def load_image_file(path: str | Path, max_size: int = 40) -> np.ndarray:
    return _load_rgba(path, max_size)


def load_emoji(emoji: str, max_size: int = 40, cache_dir: Path | None = None) -> np.ndarray:
    """Download (and cache) a Twemoji / Noto emoji PNG."""
    cache_dir = Path(cache_dir or ASSETS_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    code = _emoji_code(emoji)
    dest = cache_dir / f"{code}.png"

    if not dest.exists():
        errors = []
        for url in (_TWEMOJI_URL.format(code=code),) + tuple(
            u.format(code=code) for u in _NOTO_URLS
        ):
            try:
                _download(url, dest)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError(
                "could not download emoji image; tried:\n  " + "\n  ".join(errors)
            )
    return _load_rgba(dest, max_size)


def _procedural(name: str, size: int = 40) -> np.ndarray:
    """Draw a simple shape, supersampled 4x for antialiasing."""
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = {
        "circle": (66, 133, 244, 255),
        "ring": (219, 68, 55, 255),
        "plus": (15, 157, 88, 255),
        "star": (244, 180, 0, 255),
        "heart": (234, 67, 53, 255),
    }
    color = colors.get(name)
    if color is None:
        raise ValueError(f"unknown procedural target {name!r}")

    if name == "circle":
        pad = 2 * scale
        draw.ellipse((pad, pad, s - pad, s - pad), fill=color)
    elif name == "ring":
        pad = 2 * scale
        width = 6 * scale
        draw.ellipse((pad, pad, s - pad, s - pad), outline=color, width=width)
    elif name == "plus":
        w = s // 5
        c = s // 2
        draw.rectangle((c - w // 2, s // 6, c + w // 2, s - s // 6), fill=color)
        draw.rectangle((s // 6, c - w // 2, s - s // 6, c + w // 2), fill=color)
    elif name == "star":
        cx = cy = s / 2
        r_out, r_in = s * 0.45, s * 0.18
        pts = []
        for i in range(10):
            r = r_out if i % 2 == 0 else r_in
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, fill=color)
    elif name == "heart":
        # Two circles plus a triangle, scaled to the canvas.
        r = s * 0.27
        y = s * 0.34
        draw.ellipse((s * 0.5 - 2 * r, y - r, s * 0.5, y + r), fill=color)
        draw.ellipse((s * 0.5, y - r, s * 0.5 + 2 * r, y + r), fill=color)
        draw.polygon(
            [(s * 0.5 - 2 * r + 0.05 * s, y + 0.35 * r),
             (s * 0.5 + 2 * r - 0.05 * s, y + 0.35 * r),
             (s * 0.5, s * 0.92)],
            fill=color,
        )
    else:
        raise ValueError(f"unknown procedural target {name!r}")

    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:]
    return arr


def load_target(
    spec: str,
    max_size: int = 40,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Load a target from a spec string.

    Supported specs::

        emoji:🦎          an emoji glyph (downloaded and cached)
        emoji:1f98e       an emoji by codepoint
        image:path.png    any local image file
        shape:heart       a built-in procedural shape
                          (circle, ring, plus, star, heart)
    """
    if spec.startswith("emoji:"):
        value = spec[len("emoji:") :]
        if all(c in "0123456789abcdef-" for c in value.lower()):
            emoji = "".join(chr(int(c, 16)) for c in value.split("-"))
        else:
            emoji = value
        return load_emoji(emoji, max_size=max_size, cache_dir=cache_dir)
    if spec.startswith("image:"):
        return load_image_file(spec[len("image:") :], max_size=max_size)
    if spec.startswith("shape:"):
        return _procedural(spec[len("shape:") :], size=max_size)
    raise ValueError(
        f"unrecognised target spec {spec!r}; expected emoji:..., image:... or shape:..."
    )


def prepare_target(target_rgba: np.ndarray, padding: int) -> torch.Tensor:
    """Pad an ``[H, W, 4]`` target and return a ``[1, 4, H', W']`` tensor."""
    arr = np.asarray(target_rgba, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"target must be [H, W, 4], got {arr.shape}")
    padded = np.pad(arr, ((padding, padding), (padding, padding), (0, 0)))
    # HWC -> CHW -> BCHW
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def make_seed(height: int, width: int, channel_n: int = 16) -> torch.Tensor:
    """A single seed cell in the middle with RGB=0 and all other channels=1."""
    seed = torch.zeros(channel_n, height, width)
    seed[3:, height // 2, width // 2] = 1.0
    return seed
