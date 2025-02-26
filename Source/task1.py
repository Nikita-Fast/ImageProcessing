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


def rescale(data: np.ndarray, OldMax, OldMin, NewMax, NewMin):
    OldRange = OldMax - OldMin
    NewRange = NewMax - NewMin
    return (((data - OldMin) * NewRange) / OldRange) + NewMin


def picewise_linear_transform(img, points, show_plot=False):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    f = interpolate.interp1d(x, y)
    lut = np.array([f(r) for r in range(256)], dtype=float)
    if show_plot:
        plt.plot(lut)
        plt.show()
    return lut[img]


def hist(img, show_plot=False):
    histogram = np.zeros(256, dtype=int)
    unique, counts = np.unique(img, return_counts=True)
    histogram[unique] = counts

    if show_plot:
        plt.stem(histogram)
        plt.show()
    return histogram


def hist_eq(img):
    counts = hist(img, show_plot=True)
    lut = np.cumsum(counts) * (255 / (img.shape[0]*img.shape[1]))
    # plt.plot(unique, lut)
    # plt.show()
    return lut[img]


if __name__ == '__main__':
    # img = pic_to_array('./../Images/xray2.webp')
    # Image.fromarray(img).show()
    #
    # neg_img = negative_transform(img)
    # Image.fromarray(neg_img).show()

    # img = pic_to_array('./../Images/dark.jpg')
    # Image.fromarray(img).show()
    # log_img = log_transform(img, 23, show_plot=True)
    # Image.fromarray(log_img).show()

    # img = pic_to_array('./../Images/rev_log_exp.jpg')
    # Image.fromarray(img).show()
    # rev_log_img = reverse_log_transform(img, 80, show_plot=True)
    # Image.fromarray(rev_log_img).show()

    # img = pic_to_array('./../Images/gamma_example.PNG')
    # Image.fromarray(img).show()
    # gamma_img = gamma_correction(img, 1, gamma=0.7, show_plot=True)
    # Image.fromarray(gamma_img).show()

    # img = pic_to_array('./../Images/low_contrst_example.png')
    # Image.fromarray(img).show()
    # p1 = (0, 0)
    # p2 = (80, 20)
    # p3 = (180, 230)
    # p4 = (255, 255)
    # points = [p1,p2,p3,p4]
    # x = [p[0] for p in points]
    # y = [p[1] for p in points]
    # res_img = picewise_linear_transform(img, points, show_plot=True)
    # Image.fromarray(res_img).show()

