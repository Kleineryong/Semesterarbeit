import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import warnings
import openpyxl
import multiprocessing as mp
from scipy.constants import c, h, k
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from scipy.interpolate import interp1d
from joblib import Parallel, delayed


def main():
    t = np.linspace(500, 2000)
    wl = np.linspace(500, 1000)
    x, y = np.meshgrid(t, wl)

    intensity3d = black_body_radiation(x, (y * 1e-9))
    intensity = black_body_radiation(1000, (wl * 1e-9))

    # fig_1 = plt.figure()
    # ax = fig_1.add_subplot(111, projection='3d')
    # ax.scatter(x, y, intensity3d, s=5, c='g', marker='1')
    # ax.set_xlabel('temperature')
    # ax.set_ylabel('wavelength')
    # ax.set_zlabel('intensity')

    # plt.plot(wl, intensity)
    # plt.xlabel("Wavelength[nm]")
    # plt.ylabel('Intensity [' +r'$Wm^{-3}sr^{-1}m^{-1}$' + ']')
    # plt.savefig('black_body_radiation_1000k.jpg')
    # plt.clf()
    # plt.close(fig_1)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    lin_bb = ax1.plot(wl, intensity, 'r-', label='Black body intensity')

    lin_inten = ax1.plot(wl, intensity * emi(wl*1e-9), 'g-', label='Actual intensity' )

    lin_emi = ax2.plot(wl, emi(wl * 1e-9), 'b-', label='Emissivity')

    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Intensity[' + r'$Wm^{-2}sr^{-1}m^{-1}$' + ']')
    ax2.set_ylabel('emissivity')
    ax2.set_ylim(0, 1)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # lines = [lin_bb, lin_inten, lin_emi]
    # print(lines[1][0].get_label())

    # # labels = [lines[i][0].get_label() for i in [0, 1, 2]]
    # labels = ['Black body intensity', 'Actual intensity', 'Emissivity']
    # ax1.legend(lines, labels, loc='upper left')

    # plt.show()
    plt.savefig('real_radiation.jpg')
    return 0


def black_body_radiation(temperature, wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    return param1/(wavelength**5)/(np.exp(param2/(wavelength*temperature))-1)


def emi(wl):
    wl_rel = (wl - 500e-9) / (1000e-9 - 500e-9)
    return 0.5 - wl_rel * 0.1


if 1:
    main()
