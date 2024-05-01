from PIL import Image
import numpy as np

def pil_loader(path, crop=True):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if crop:
            img = img.crop((0, 0, 512, 487))

        return img.convert('RGB')