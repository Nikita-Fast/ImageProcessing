from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate


def pic_to_array(path: str):
    return np.array(Image.open(path).convert('L'))


def array_to_pic(arr, name):
    im = Image.fromarray(arr)
    im.save(f"{name}.jpeg")