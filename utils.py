import numpy as np
from PIL import Image


def expand2square(pil_img: Image, background_color: tuple):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def resize_square_image(
    img: Image, width: int = 640, background_color: tuple = (0, 0, 0)
):
    if img.mode != "RGB":
        return None
    img = expand2square(img, (0, 0, 0))
    img = img.resize((width, width))

    return img


def crop_center_square(img: Image):
    if img.mode != "RGB":
        return None

    width, height = img.size

    if width > height:
        margin = (width - height) // 2
        left = margin
        right = width - margin
        top = 0
        bottom = height

    else:
        margin = (height - width) // 2
        left = 0
        right = width
        top = margin
        bottom = height - margin

    img = img.crop((left, top, right, bottom))

    return img
