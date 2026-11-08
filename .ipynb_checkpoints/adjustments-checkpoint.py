"""
Main public functions
---------------------
basic_adjustments(image, **settings)
apply_tone_curve(image, points, channel="rgb")
get_tone_curve_preset(name)
hsl_color_mix(image, adjustments)
color_grade_3way(image, shadows, midtones, highlights, blending=50, balance=0)
develop_image(image, basic=None, curve_points=None, hsl=None, color_grading=None)

Slider conventions
------------------
Basic panel:
    temperature, tint, contrast, highlights, shadows, whites, blacks,
    texture, clarity, dehaze, vibrance, saturation: generally -100..+100
    exposure: EV, generally -5..+5

HSL mix:
    H: -100..+100 (mapped to -180..+180 degrees)
    S and L: -100..+100

Color grading wheel dictionaries : {"hue": 0..360, "saturation": 0..100, "luminance": -100..+100}
Tone curve points: list of (input, output) pairs on the 0..255 scale, ordered by input.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8

def _as_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("Expected a grayscale (H, W) or RGB (H, W, 3) image.")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError("Color images must have exactly 3 RGB channels.")
    if arr.size == 0:
        raise ValueError("Image is empty.")
    return np.clip(arr.astype(np.float32, copy=True), 0.0, 255.0)


def _is_rgb(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 3


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2990 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2])


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _box_blur_2d(channel: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return channel.astype(np.float32, copy=True)

    src = np.asarray(channel, dtype=np.float32)
    pad = int(radius)
    padded = np.pad(src, ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)

    size = 2 * pad + 1
    h, w = src.shape
    y0 = np.arange(h)
    y1 = y0 + size
    x0 = np.arange(w)
    x1 = x0 + size

    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return sums / float(size * size)


def _box_blur_image(image: np.ndarray, radius: int) -> np.ndarray:
    if image.ndim == 2:
        return _box_blur_2d(image, radius)
    return np.stack(
        [_box_blur_2d(image[..., c], radius) for c in range(3)],
        axis=2,
    )


def _adjust_saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0:
        return rgb
    lum = _luminance(rgb)[..., None]
    # -100 -> grayscale; +100 -> 2x chroma.
    factor = max(0.0, 1.0 + float(amount) / 100.0)
    return lum + (rgb - lum) * factor


def _apply_temperature_tint(rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    out = rgb.copy()
    t = np.clip(float(temperature), -100.0, 100.0) / 100.0
    g = np.clip(float(tint), -100.0, 100.0) / 100.0

    out[..., 0] *= 1.0 + 0.12 * t
    out[..., 2] *= 1.0 - 0.12 * t

    out[..., 0] *= 1.0 + 0.045 * g
    out[..., 1] *= 1.0 - 0.075 * g
    out[..., 2] *= 1.0 + 0.045 * g
    return out


def basic_adjustments(
    image: np.ndarray,
    *,
    temperature: float = 0.0,
    tint: float = 0.0,
    exposure: float = 0.0,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
    texture: float = 0.0,
    clarity: float = 0.0,
    dehaze: float = 0.0,
    vibrance: float = 0.0,
    saturation: float = 0.0,
) -> np.ndarray:
    """Apply Lightroom-style Basic panel adjustments.
    """
    out = _as_float_image(image)
    is_rgb = _is_rgb(out)

    if is_rgb and (temperature != 0 or tint != 0):
        out = _apply_temperature_tint(out, temperature, tint)

    if exposure != 0:
        out *= np.float32(2.0 ** float(np.clip(exposure, -5.0, 5.0)))

    if contrast != 0:
        factor = 2.0 ** (float(np.clip(contrast, -100.0, 100.0)) / 50.0)
        out = (out - 128.0) * factor + 128.0

    lum = _luminance(out) if is_rgb else out
    norm = np.clip(lum / 255.0, 0.0, 1.0)

    if shadows != 0:
        shadow_mask = 1.0 - _smoothstep(0.12, 0.72, norm)
        out += (float(shadows) / 100.0 * 65.0) * shadow_mask[..., None] if is_rgb else (float(shadows) / 100.0 * 65.0 * shadow_mask)

    if highlights != 0:
        highlight_mask = _smoothstep(0.30, 0.92, norm)
        out += (-float(highlights) / 100.0 * 65.0) * (highlight_mask[..., None] if is_rgb else highlight_mask)

    if blacks != 0:
        black_mask = 1.0 - _smoothstep(0.0, 0.38, norm)
        out += (float(blacks) / 100.0 * 45.0) * (black_mask[..., None] if is_rgb else black_mask)

    if whites != 0:
        white_mask = _smoothstep(0.62, 1.0, norm)
        out += (float(whites) / 100.0 * 45.0) * (white_mask[..., None] if is_rgb else white_mask)

    out = np.clip(out, 0.0, 255.0)

    if texture != 0 or clarity != 0:
        if is_rgb:
            base_lum = _luminance(out)
            if texture != 0:
                fine_base = _box_blur_2d(base_lum, 1)
                detail = base_lum - fine_base
                strength = float(np.clip(texture, -100.0, 100.0)) / 100.0
                out += (detail * 0.65 * strength)[..., None]
            if clarity != 0:
                current_lum = _luminance(out)
                local_base = _box_blur_2d(current_lum, 6)
                local_detail = current_lum - local_base
                strength = float(np.clip(clarity, -100.0, 100.0)) / 100.0
                out += (local_detail * 0.55 * strength)[..., None]
        else:
            if texture != 0:
                detail = out - _box_blur_2d(out, 1)
                out += detail * (0.65 * float(np.clip(texture, -100.0, 100.0)) / 100.0)
            if clarity != 0:
                detail = out - _box_blur_2d(out, 6)
                out += detail * (0.55 * float(np.clip(clarity, -100.0, 100.0)) / 100.0)

    if dehaze != 0:
        strength = float(np.clip(dehaze, -100.0, 100.0)) / 100.0
        pivot = 128.0
        factor = 1.0 + 0.45 * strength
        out = (out - pivot) * factor + pivot
        if is_rgb and strength > 0:
            out = _adjust_saturation(out, 12.0 * strength)

    if is_rgb:
        if vibrance != 0:
            channel_max = np.max(out, axis=2)
            channel_min = np.min(out, axis=2)
            current_sat = (channel_max - channel_min) / np.maximum(channel_max, _EPS)
            selective = (1.0 - current_sat) ** 1.5
            lum = _luminance(out)[..., None]
            factor = 1.0 + (float(np.clip(vibrance, -100.0, 100.0)) / 100.0) * selective
            out = lum + (out - lum) * factor[..., None]

        if saturation != 0:
            out = _adjust_saturation(out, float(np.clip(saturation, -100.0, 100.0)))

    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _validate_curve_points(points) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("Tone curve points must contain at least two (x, y) pairs.")
    if np.any(~np.isfinite(pts)):
        raise ValueError("Tone curve points must be finite numbers.")
    pts = np.clip(pts, 0.0, 255.0)
    order = np.argsort(pts[:, 0], kind="stable")
    pts = pts[order]
    x = pts[:, 0]
    y = pts[:, 1]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Tone curve input coordinates (x) must be unique and increasing.")
    return x, y


def apply_tone_curve(image: np.ndarray, points, channel: str = "rgb") -> np.ndarray:
    """
    channel='rgb' applies the same curve independently to R, G and B (or to
    the grayscale channel). 
    """
    out = _as_float_image(image)
    x, y = _validate_curve_points(points)
    lut = np.interp(np.arange(256, dtype=np.float32), x, y).astype(np.float32)

    channel = channel.lower()
    if channel not in {"rgb", "red", "green", "blue"}:
        raise ValueError("channel must be 'rgb', 'red', 'green', or 'blue'.")

    if out.ndim == 2:
        if channel != "rgb":
            raise ValueError("Individual color channels are only available for RGB images.")
        indices = np.clip(np.rint(out), 0, 255).astype(np.uint8)
        return lut[indices]

    if channel == "rgb":
        indices = np.clip(np.rint(out), 0, 255).astype(np.uint8)
        return lut[indices]

    channel_index = {"red": 0, "green": 1, "blue": 2}[channel]
    indices = np.clip(np.rint(out[..., channel_index]), 0, 255).astype(np.uint8)
    out[..., channel_index] = lut[indices]
    return out


def get_tone_curve_preset(name: str):
    """Return 0..255 control points for a named composite tone curve."""
    presets = {
        "linear": [
            (0, 0), (255, 255)
        ],
        "subtle_s": [
            (0, 0), (48, 40), (128, 128), (208, 218), (255, 255)
        ],
        "strong_s": [
            (0, 0), (48, 28), (128, 128), (208, 230), (255, 255)
        ],
        "matte": [
            (0, 18), (48, 55), (128, 132), (208, 218), (255, 248)
        ],
        "faded": [
            (0, 24), (64, 76), (128, 136), (192, 205), (255, 248)
        ],
    }
    key = name.lower().strip().replace(" ", "_").replace("-", "_")
    if key not in presets:
        raise ValueError(f"Unknown tone curve preset: {name!r}. Choose from {', '.join(presets)}.")
    return presets[key].copy()

def _rgb_to_hsl(rgb: np.ndarray):
    arr = np.clip(rgb / 255.0, 0.0, 1.0)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.max(arr, axis=2)
    cmin = np.min(arr, axis=2)
    delta = cmax - cmin
    light = (cmax + cmin) / 2.0

    sat = np.zeros_like(light)
    denom = 1.0 - np.abs(2.0 * light - 1.0)
    np.divide(delta, denom, out=sat, where=(delta > _EPS) & (denom > _EPS))

    hue = np.zeros_like(light)
    nonzero = delta > _EPS
    rmax = nonzero & (cmax == r)
    gmax = nonzero & (cmax == g)
    bmax = nonzero & (cmax == b)

    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue = (hue * 60.0) % 360.0
    return hue, np.clip(sat, 0.0, 1.0), np.clip(light, 0.0, 1.0)


def _hsl_to_rgb(hue: np.ndarray, sat: np.ndarray, light: np.ndarray) -> np.ndarray:
    h = (hue % 360.0) / 60.0
    s = np.clip(sat, 0.0, 1.0)
    l = np.clip(light, 0.0, 1.0)

    chroma = (1.0 - np.abs(2.0 * l - 1.0)) * s
    x = chroma * (1.0 - np.abs((h % 2.0) - 1.0))
    m = l - chroma / 2.0

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    masks = [
        (h >= 0) & (h < 1),
        (h >= 1) & (h < 2),
        (h >= 2) & (h < 3),
        (h >= 3) & (h < 4),
        (h >= 4) & (h < 5),
        (h >= 5) & (h < 6),
    ]
    values = [
        (chroma, x, np.zeros_like(h)),
        (x, chroma, np.zeros_like(h)),
        (np.zeros_like(h), chroma, x),
        (np.zeros_like(h), x, chroma),
        (x, np.zeros_like(h), chroma),
        (chroma, np.zeros_like(h), x),
    ]
    for mask, (rv, gv, bv) in zip(masks, values):
        r[mask], g[mask], b[mask] = rv[mask], gv[mask], bv[mask]

    return np.stack((r + m, g + m, b + m), axis=2) * 255.0


_HUE_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 275.0,
    "magenta": 315.0,
}


def _hue_range_mask(hue: np.ndarray, center: float, width: float = 38.0) -> np.ndarray:
    """Soft circular hue selection; handles the red range across 0/360."""
    distance = np.abs((hue - center + 180.0) % 360.0 - 180.0)
    t = np.clip(1.0 - distance / width, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hsl_color_mix(image: np.ndarray, adjustments: dict) -> np.ndarray:
    """Adjust selected HSL color ranges.
    """
    out = _as_float_image(image)
    if not _is_rgb(out):
        raise ValueError("HSL Color Mix requires an RGB image.")
    if not adjustments:
        return out

    hue, sat, light = _rgb_to_hsl(out)
    hue_delta = np.zeros_like(hue)
    sat_delta = np.zeros_like(sat)
    light_delta = np.zeros_like(light)

    for name, values in adjustments.items():
        key = name.lower().strip()
        if key not in _HUE_CENTERS:
            raise ValueError(f"Unknown HSL color range: {name!r}.")
        values = values or {}
        mask = _hue_range_mask(hue, _HUE_CENTERS[key])

        h_value = float(values.get("h", values.get("H", 0.0)))
        s_value = float(values.get("s", values.get("S", 0.0)))
        l_value = float(values.get("l", values.get("L", 0.0)))

        hue_delta += mask * (np.clip(h_value, -100.0, 100.0) * 1.8)
        sat_delta += mask * (np.clip(s_value, -100.0, 100.0) / 100.0)
        light_delta += mask * (np.clip(l_value, -100.0, 100.0) / 100.0 * 0.25)

    new_hue = (hue + hue_delta) % 360.0
    new_sat = np.clip(sat * (1.0 + sat_delta), 0.0, 1.0)
    new_light = np.clip(light + light_delta, 0.0, 1.0)

    return np.clip(_hsl_to_rgb(new_hue, new_sat, new_light), 0.0, 255.0).astype(np.float32)


def _hue_to_rgb_unit(hue_degrees: float) -> np.ndarray:
    h = (float(hue_degrees) % 360.0) / 60.0
    c = 1.0
    x = 1.0 - abs((h % 2.0) - 1.0)
    if h < 1:
        rgb = (c, x, 0.0)
    elif h < 2:
        rgb = (x, c, 0.0)
    elif h < 3:
        rgb = (0.0, c, x)
    elif h < 4:
        rgb = (0.0, x, c)
    elif h < 5:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    return np.asarray(rgb, dtype=np.float32)


def color_grade_3way(
    image: np.ndarray,
    shadows: dict | None = None,
    midtones: dict | None = None,
    highlights: dict | None = None,
    *,
    blending: float = 50.0,
    balance: float = 0.0,
) -> np.ndarray:
    """Apply shadow, midtone and highlight color-wheel grading.
    """
    out = _as_float_image(image)
    if not _is_rgb(out):
        raise ValueError("3-Way Color Grading requires an RGB image.")

    shadows = shadows or {}
    midtones = midtones or {}
    highlights = highlights or {}

    if not (shadows or midtones or highlights):
        return out

    lum = np.clip(_luminance(out) / 255.0, 0.0, 1.0)
    blend = np.clip(float(blending), 0.0, 100.0) / 100.0
    bal = np.clip(float(balance), -100.0, 100.0) / 100.0

    # Shift tonal zones to favor one side of the tonal range.
    adjusted_lum = np.clip(lum + bal * 0.22, 0.0, 1.0)

    # Wider Gaussian zones create more overlap as blending increases.
    sigma = 0.16 + 0.15 * blend
    centers = (0.18, 0.50, 0.82)
    weights = [
        np.exp(-0.5 * ((adjusted_lum - centers[0]) / sigma) ** 2),
        np.exp(-0.5 * ((adjusted_lum - centers[1]) / sigma) ** 2),
        np.exp(-0.5 * ((adjusted_lum - centers[2]) / sigma) ** 2),
    ]
    weight_sum = weights[0] + weights[1] + weights[2] + _EPS
    weights = [w / weight_sum for w in weights]

    out = out.copy()
    for mask, wheel in zip(weights, (shadows, midtones, highlights)):
        if not wheel:
            continue
        hue = float(wheel.get("hue", 0.0)) % 360.0
        sat = np.clip(float(wheel.get("saturation", 0.0)), 0.0, 100.0) / 100.0
        light = np.clip(float(wheel.get("luminance", 0.0)), -100.0, 100.0) / 100.0

        if sat > 0:
            tint_color = _hue_to_rgb_unit(hue) * 255.0
            tint_delta = (tint_color - 127.5) * (0.30 * sat)
            out += mask[..., None] * tint_delta[None, None, :]

        if light != 0:
            out += (mask * light * 32.0)[..., None]

    return np.clip(out, 0.0, 255.0).astype(np.float32)


def develop_image(
    image: np.ndarray,
    *,
    basic: dict | None = None,
    curve_points=None,
    hsl: dict | None = None,
    color_grading: dict | None = None,
) -> np.ndarray:
    out = _as_float_image(image)

    if basic:
        out = basic_adjustments(out, **basic)

    if curve_points is not None:
        out = apply_tone_curve(out, curve_points, channel="rgb")

    if hsl:
        if _is_rgb(out):
            out = hsl_color_mix(out, hsl)

    if color_grading:
        if _is_rgb(out):
            grading = dict(color_grading)
            blending = grading.pop("blending", 50.0)
            balance = grading.pop("balance", 0.0)
            out = color_grade_3way(
                out,
                shadows=grading.get("shadows"),
                midtones=grading.get("midtones"),
                highlights=grading.get("highlights"),
                blending=blending,
                balance=balance,
            )

    return np.clip(out, 0.0, 255.0).astype(np.float32)
