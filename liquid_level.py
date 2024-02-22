import cv2 as cv
from PIL.Image import Image, fromarray
import numpy as np
import torchvision
import torchvision.transforms.functional as F

def _binarize(img: Image) -> Image:
    flat_img = np.array(img).sum(-1)
    new_img = np.zeros_like(img)
    threshold = flat_img.mean()
    new_img[flat_img > threshold] = 255
    return fromarray(new_img)

def _preprocess(img: Image) -> Image:
    img = F.adjust_brightness(img, 1.5)
    img = F.adjust_saturation(img, 0)
    img = _binarize(img)
    return img

def _rowsum(img: Image) -> np.ndarray:
    img_n = np.array(img)
    p = 7
    v = img_n.shape[1]//p
    rs = (img_n[:, v*(p//2-1):v*(p//2), :].mean(-1) / 255).sum(-1)
    return rs

def _find_level(res: np.ndarray, w: int) -> int:
    level = np.where(res <= w // 7 // 7)[0][0].item()
    return level

def get_water_level(img: Image) -> int:
    img_p = _preprocess(img)
    img_p.save("_pp.jpg")
    rs = _rowsum(img_p)

    s = rs.shape[0] // 5
    if rs[s*4:].var() > 10 and rs[s*3:s*4].var() > 10 and rs[s*2:s*3].var() > 10:
        return img_p.size[1]
    else:
        return _find_level(rs, img_p.size[0])