from scipy.constants import c, h, k
import numpy as np
import matplotlib.pyplot as plt


def wiens_law():
    wl_set = np.linspace(150, 2000, 2000)
    t = [2000, 2500, 3000, 3500, 4000, 4500]
    fig = plt.figure(figsize=(8, 6), dpi=100)
    for i in t:
        plt.plot(wl_set, black_body_radiation(i, wl_set*1e-9), label=str(i) + 'K')

    plt.legend(fontsize=15)
    plt.xlabel('wavelength[nm]', fontsize=15)
    plt.ylabel('Intensity[' + r'$Wm^{-2}sr^{-1}m^{-1}$' + ']', fontsize=15)
    plt.tick_params(axis='x', labelsize=15)  # 设置x轴刻度数字的大小为10
    plt.tick_params(axis='y', labelsize=15)  # 设置y轴刻度数字的大小为10
    plt.tight_layout()
    plt.savefig('wiens_law.jpg')
    # plt.show()


def black_body_radiation(temperature, wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    return param1/(wavelength**5)/(np.exp(param2/(wavelength*temperature))-1)

if 1:
    wiens_law()