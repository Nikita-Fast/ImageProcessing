from PIL import Image
import numpy as np


def negative(img):
    # s=L-1-r
    lut = np.array([255-r for r in range(256)], dtype=np.uint8)
    return lut[img]


def log_transfiguration(img, c):
    # s = c * log(1+r)
    lut = np.array([c * np.log(1+r) for r in range(256)], dtype=np.uint8)
    print(1)
    return lut[img]


def rescale(data: np.ndarray, OldMax, OldMin, NewMax, NewMin):
    OldRange = OldMax - OldMin
    NewRange = NewMax - NewMin
    return (((data - OldMin) * NewRange) / OldRange) + NewMin


if __name__ == '__main__':
    img = np.array(Image.open('./../Images/IMG20250203105039.png').convert('RGB'))
    # neg_img = negative(img)
    # Image.fromarray(neg_img).show()

    # log_img = log_transfiguration(img, 3)
    # Image.fromarray(log_img).show()

