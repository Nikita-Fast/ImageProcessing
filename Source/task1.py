from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate

from Source.task2 import local_hist
from Source.utils import pic_to_array, array_to_pic


def negative_transform(img):
    # s=L-1-r
    lut = np.array([255-r for r in range(256)], dtype=np.uint8)
    return lut[img]


def log_transform(img, c=46, show_plot=False):
    lut = np.array([c * np.log(1+r) for r in range(256)], dtype=np.uint8)
    lut = np.array(lut / np.max(lut) * 255, dtype=np.uint8)
    if show_plot:
        plt.plot(lut)
        plt.show()
    return lut[img]


def reverse_log_transform(img, c=46, show_plot=False):
    lut = np.array([np.exp(r/c)-1 for r in range(256)], dtype=np.uint8)
    lut = np.array(lut / np.max(lut) * 255, dtype=np.uint8)
    if show_plot:
        plt.plot(lut)
        plt.show()
    return lut[img]


def gamma_correction(img, c, gamma, show_plot=False):
    lut = np.array([c * r**gamma for r in range(256)], dtype=float)
    lut = np.array(lut / np.max(lut) * 255, dtype=np.uint8)
    if show_plot:
        plt.plot(lut)
        plt.show()
    return lut[img]


def picewise_linear_transform(img, points, show_plot=False):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    f = interpolate.interp1d(x, y)
    lut = np.array([f(r) for r in range(256)], dtype=np.uint8)
    if show_plot:
        plt.plot(lut)
        plt.show()
    return lut[img]


if __name__ == '__main__':
    img = pic_to_array(r"D:\Projects\pythonProject\pythonProject\ImageProcessing\Images\1\low_contrst_example.png")
    res = picewise_linear_transform(img, [(0,0), (80,20), (160,240), (255,255)])

    array_to_pic(res, "../res_img/1/picewise_linear_transform")


