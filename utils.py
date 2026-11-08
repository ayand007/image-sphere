from PIL import Image
import numpy as np


def load_image(file):
    
    if hasattr(file, "seek"):
        file.seek(0)

    img = Image.open(file)
    img.load()

    if img.mode in ("1", "L", "I", "F"):
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

        return arr, gray, False

    # Preserve transparency when converting to RGB by
    # compositing transparent areas over a white background.
    if "A" in img.getbands() or "transparency" in img.info:
        rgba = img.convert("RGBA")

        background = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255)
        )

        img = Image.alpha_composite(
            background,
            rgba
        ).convert("RGB")

    else:
        
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.float32)

    return arr, img, True


def to_image(arr):
    """Convert a NumPy array into a displayable PIL image."""

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")

    return Image.fromarray(arr, mode="RGB")