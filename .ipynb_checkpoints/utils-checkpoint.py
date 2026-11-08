from PIL import Image
import numpy as np

def load_image(file):
    img = Image.open(file)
    if img.mode == "RGB":
        return np.array(img).astype(float), img, True
    else:
        img = img.convert("L")
        return np.array(img).astype(float), img, False

def to_image(arr):
    return Image.fromarray(arr.astype(np.uint8))