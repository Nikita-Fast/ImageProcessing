from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def pic_to_array(path: str):
    return np.array(Image.open(path).convert('L'))

def negative_transform(img):
    # s=L-1-r
    lut = np.array([255-r for r in range(256)], dtype=np.uint8)
    return lut[img]


def log_transform(img, c=46):
    lut = np.array([c * np.log(1+r) for r in range(256)], dtype=np.uint8)
    lut = np.array(lut / np.max(lut) * 255, dtype=np.uint8)
    return lut[img]


def reverse_log_transform(img, c=46):
    lut = np.array([np.exp(r/c)-1 for r in range(256)], dtype=np.uint8)
    lut = np.array(lut / np.max(lut) * 255, dtype=np.uint8)
    return lut[img]


def gamma_correction(img, c, gamma):
    lut = np.array([c * r**gamma for r in range(256)], dtype=float)
    lut = np.array(lut / np.max(lut) * 255, dtype=np.uint8)
    return lut[img]


def rescale(data: np.ndarray, OldMax, OldMin, NewMax, NewMin):
    OldRange = OldMax - OldMin
    NewRange = NewMax - NewMin
    return (((data - OldMin) * NewRange) / OldRange) + NewMin


def picewise_linear_transform(img, points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    f = interpolate.interp1d(x, y)
    lut = np.array([f(r) for r in range(256)], dtype=float)
    return lut[img]


if __name__ == '__main__':
    img = pic_to_array('./../Images/too_light.png')
    # Image.fromarray(img).show()

    # neg_img = negative_transform(img)
    # Image.fromarray(neg_img).show()
    #
    # log_img = log_transform(img, 46)
    # Image.fromarray(log_img).show()

    # rev_log_img = reverse_log_transform(img, 46)
    # Image.fromarray(rev_log_img).show()

    # gamma_img = gamma_correction(img, 1, gamma=2.2)
    # Image.fromarray(gamma_img).show()


    p1 = (0, 20)
    p2 = (50, 20)
    p3 = (50, 120)
    p4 = (100, 120)
    p5 = (100, 20)
    p6 = (255, 20)
    points = [p1,p2,p3,p4,p5,p6]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    xui = picewise_linear_transform(img, points)
    Image.fromarray(xui).show()

