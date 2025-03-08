import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from Source.utils import pic_to_array, array_to_pic


def hist(img, L):
    histogram = np.zeros(L, dtype=int)
    unique, counts = np.unique(img, return_counts=True)
    histogram[unique] = counts
    return histogram


def plot_hist(images: list, labels: list, L=256):
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        y = hist(img, L)
        x = np.nonzero(y)[0]
        plt.stem(x, y[x], linefmt=f'C{i}-', markerfmt=f'C{i}o', label=labels[i])
        plt.legend()
    plt.show()


def get_intensity_probabilities(img, L):
    h = hist(img, L)
    n = img.shape[0] * img.shape[1]
    p = h / n
    return p


def get_hist_eq_lut(img, L):
    p = get_intensity_probabilities(img, L)
    lut = (L - 1) * np.cumsum(p)
    lut = np.array(lut, np.uint8)
    return lut


def hist_eq(img, L):
    lut = get_hist_eq_lut(img, L)
    return lut[img]


def local_hist(img, func, w_size=3, L=256):
    assert w_size % 2 == 1

    pad_width = w_size // 2
    v = sliding_window_view(np.pad(img, [pad_width, pad_width]), (w_size, w_size))
    res = np.zeros(img.shape, dtype=np.uint8)
    for i, j in np.ndindex(img.shape):
        print(f"{i}/{img.shape[0]}")
        r = img[i, j]
        s = func(v[i, j], r)
        res[i, j] = s

    return res


if __name__ == '__main__':
    img = pic_to_array(r"D:\Projects\pythonProject\pythonProject\ImageProcessing\Images\2\Fig0326(a)(embedded_square_noisy_512).tif")

    def window_hist_eq(w, r):
        lut = get_hist_eq_lut(w, 256)
        return lut[r]

    res = local_hist(img, func=window_hist_eq, w_size=3)
    array_to_pic(res, "../res_img/2/Fig0326(a)")
    plot_hist([img, res], labels=["img", "global_hist_eq"])




    img = pic_to_array(r"D:\Projects\pythonProject\pythonProject\ImageProcessing\Images\2\Fig0327(a)(tungsten_original).tif")
    x = hist_eq(img, L=256)

    g_mean = np.mean(img)
    g_std = np.std(img)

    def enhance_dark_remain_light(w, r):
        local_mean = np.mean(w)
        local_std = np.std(w)

        k0, k1, k2 = 0.4, 0.02, 0.4
        E = 4

        if (local_mean <= k0*g_mean) and (k1*g_std <= local_std <= k2*g_std):
            return E * r
        else:
            return r

    y = local_hist(img, func=enhance_dark_remain_light, w_size=3)
    array_to_pic(x, "../res_img/2/Fig0327(a)_global")
    array_to_pic(y, "../res_img/2/Fig0327(a)_local")

    plot_hist([img, x, y], labels=["img", "global", "local"])




