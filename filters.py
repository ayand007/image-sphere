import numpy as np

def convolve(img, kernel):
    # h, w = img.shape
    # kh, kw = kernel.shape
    # pad = kh // 2

    # padded = np.pad(img, pad, mode='edge')
    # out = np.zeros_like(img, dtype=float)

    # for i in range(h):
    #     for j in range(w):
    #         region = padded[i:i+kh, j:j+kw]
    #         out[i, j] = np.sum(region * kernel)

    # return np.clip(out, 0, 255)
    kh, kw = kernel.shape
    pad = kh // 2

    padded = np.pad(img, pad, mode='edge')
    shape = (
        img.shape[0],
        img.shape[1],
        kh,
        kw
    )

    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1]
    )

    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=shape,
        strides=strides
    )

    out = np.einsum('ijkl,kl->ij', windows, kernel)

    return np.clip(out, 0, 255)


def brightness(img, val):
    return np.clip(img + val, 0, 255)


def contrast(img, val):
    return np.clip((img - 128) * val + 128, 0, 255)


def threshold(img, t):
    if t > 0:
        return np.where(img > t, 255, 0)
    return img


def negative(img):
    return 255 - img


def blur(img):
    kernel = np.ones((3,3)) / 9
    return convolve(img, kernel)


def sharpen(img):
    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])
    return convolve(img, kernel)


def edge_detect(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    return np.sqrt(gx**2 + gy**2)

def rotate_img(img, angle):
    if angle == 0:
        return img
    return np.rot90(img, angle // 90)

def flip_horizontal(img):
    return img[:, ::-1]

def flip_vertical(img):
    return img[::-1, :]

def _clip(img):
    return np.clip(img, 0, 255)


def _luminance(img):
    return (
        0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    )


def _saturation(img, factor):
    """
    factor:
        1.0 = unchanged
        < 1 = desaturated
        > 1 = more saturated
    """

    lum = _luminance(img)

    gray = np.stack([lum, lum, lum],axis=2)

    return gray + (img - gray) * factor


# def _contrast(img, factor):
#     return np.clip(
#         (img - 128) * factor + 128,
#         0,
#         255
#     )


def _warm(img, amount):
    """
    Add warm color cast.
    """

    result = img.copy()

    result[:, :, 0] += amount 
    result[:, :, 1] += amount * 0.30 
    result[:, :, 2] -= amount * 0.35 

    return _clip(result)


def _cool(img, amount):
    """
    Add cool blue/cyan color cast.
    """

    result = img.copy()

    result[:, :, 0] -= amount * 0.35
    result[:, :, 1] += amount * 0.15
    result[:, :, 2] += amount

    return _clip(result)


def _fade(img, amount=15):
    """
    Lift the black level slightly,
    producing a faded film appearance.
    """

    return np.clip(
        img * (255 - 2 * amount) / 255 + amount, 0, 255
    )


def _blend(original, filtered, strength):
    """
    Filter intensity.

    strength = 0 -> original
    strength = 1 -> full filter
    """

    result = (
        original * (1 - strength) + filtered * strength
    )

    return _clip(result)


def frblair_filter(img):
    """
    Frblair-style approximation.

    Characteristics:
    - warm film tone
    - soft contrast
    - slightly increased saturation
    - red/pink emphasis
    - slightly faded image
    """

    result = img.astype(float).copy()

    # Slightly soften contrast
    result = contrast(result, 0.92)

    # Warm film tone
    result = _warm(result, 8)

    # Slight pink/red emphasis
    result[:, :, 0] += 6
    result[:, :, 2] += 2

    # Slight saturation boost
    result = _saturation(result, 1.08)

    # Mild fade
    result = _fade(result, 5)

    return _clip(result)


def cinematic_filter(img):
    """
    Cinematic color grade.

    Characteristics:
    - warm highlights
    - stronger contrast
    - deeper shadows
    - teal/cyan shadows
    """

    result = img.astype(float).copy()

    lum = _luminance(result) / 255.0

    # Shadow mask
    shadow = 1.0 - lum

    # Highlight mask
    highlight = lum

    # Add teal to shadows
    result[:, :, 0] -= 12 * shadow
    result[:, :, 1] += 8 * shadow
    result[:, :, 2] += 18 * shadow

    # Add orange/warm tone to highlights
    result[:, :, 0] += 15 * highlight
    result[:, :, 1] += 6 * highlight
    result[:, :, 2] -= 8 * highlight

    # Increase contrast
    result = contrast(result, 1.10)

    # Slightly reduce saturation
    result = _saturation(result, 0.95)

    return _clip(result)


def vintage_filter(img):
    """
    Vintage film appearance.
    - warm tone
    - less saturation
    - less contrast
    - faded blacks
    """

    result = img.astype(float).copy()

    # Fade blacks
    result = _fade(result, 18)

    # Warm tone
    result = _warm(result, 12)

    # Reduce saturation
    result = _saturation(result, 0.82)

    # Slightly reduce contrast
    result = contrast(result, 0.92)

    return _clip(result)


def lofi_dusk_filter(img):
    """
    Dark/moody dusk appearance.
    - cool shadow
    - strong contrast
    - cool tone, slight magenta 
    """

    result = img.astype(float).copy()

    lum = _luminance(result) / 255.0

    shadow = 1.0 - lum
    highlight = lum

    # Darken overall image
    result *= 0.93

    # Cool shadows
    result[:, :, 0] -= 8 * shadow
    result[:, :, 1] += 3 * shadow
    result[:, :, 2] += 14 * shadow

    # Slight magenta warmth in highlights
    result[:, :, 0] += 8 * highlight
    result[:, :, 2] += 5 * highlight

    # Muted colors
    result = _saturation(result, 0.8)

    # Stronger contrast
    result = contrast(result, 1.08)

    return _clip(result)


def dreamy_filter(img):
    """
    - Soft-focus appearance
    - Warm, pinkish highlights
    - Cool violet-blue shadows
    - Faded blacks and reduced contrast
    - Muted pastel colors
    - Subtle highlight bloom

    """

    result = img.astype(np.float32).copy()

    gaussian_kernel = np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=np.float32) / 256.0

    soft = np.stack([
        convolve(result[:, :, channel], gaussian_kernel)
        for channel in range(3)
    ], axis=2)

    result = 0.86 * result + 0.14 * soft

    luminance = _luminance(result) / 255.0

    shadows = np.clip(
        (0.65 - luminance) / 0.65, 0.0, 1.0
    )

    highlights = np.clip(
        (luminance - 0.30) / 0.70, 0.0, 1.0
    ) ** 1.2

    # Warm, slightly pink highlights.
    result[:, :, 0] += 7.0 * highlights
    result[:, :, 1] += 3.0 * highlights
    result[:, :, 2] -= 1.5 * highlights

    # Subtle violet-blue tint in the shadows.
    result[:, :, 0] += 2.0 * shadows
    result[:, :, 1] += 1.0 * shadows
    result[:, :, 2] += 9.0 * shadows

    result = _clip(result)
    result = contrast(result, 0.82)

    # Lift dark areas and soften the tonal range.
    result = result * 0.90 + 18.0

    # Muted, pastel-like colors.
    result = _saturation(result, 0.90)

    result = _clip(result)

    # -------------------------------------------------
    # 5. SUBTLE HIGHLIGHT BLOOM
    # Bright regions spread a gentle glow into nearby
    # pixels, creating a hazy, dreamy atmosphere.
    # -------------------------------------------------

    bright_areas = np.clip(result - 165.0, 0, 255)

    bloom = np.stack([
        convolve(
            bright_areas[:, :, channel], gaussian_kernel
        )
        for channel in range(3)
    ], axis=2)

    result = result + 0.22 * bloom
    return _clip(result)

def teal_orange_filter(img):
    """
    Strong teal-and-orange cinematic style.
    """

    result = img.astype(float).copy()

    lum = _luminance(result) / 255.0

    shadow = 1.0 - lum
    highlight = lum

    # Teal shadows
    result[:, :, 0] -= 15 * shadow
    result[:, :, 1] += 10 * shadow
    result[:, :, 2] += 20 * shadow

    # Orange highlights
    result[:, :, 0] += 20 * highlight
    result[:, :, 1] += 8 * highlight
    result[:, :, 2] -= 10 * highlight

    result = contrast(result, 1.12)
    result = _saturation(result, 1.08)

    return _clip(result)


def color_filter(img, filter_name, strength=1.0):

    if filter_name == "None":
        return img

    if img.ndim != 3 or img.shape[2] != 3:
        return img

    original = img.astype(float)

    if filter_name == "Frblair":
        filtered = frblair_filter(original)

    elif filter_name == "Cinematic":
        filtered = cinematic_filter(original)

    elif filter_name == "Vintage Film":
        filtered = vintage_filter(original)

    elif filter_name == "Lofi Dusk":
        filtered = lofi_dusk_filter(original)

    elif filter_name == "Dreamy Glow":
        filtered = dreamy_filter(original)

    elif filter_name == "Teal & Orange":
        filtered = teal_orange_filter(original)

    else:
        filtered = original

    return _blend(original, filtered, strength)