import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def extrapolate_demo():
    emi_set = '31'
    data = read_raw_data(emi_set)
    emi_fun = interp1d(data[:, 0], data[:, 1], fill_value='extrapolate')
    wl_set = np.linspace(400, 800, 30)
    emi = emi_fun(wl_set)

    fig = plt.figure(figsize=(8, 6), dpi=400)
    plt.plot(wl_set, emi)
    plt.xlabel("Wavelength[nm]", fontsize=20)
    plt.ylabel("Emissivity", fontsize=20)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("emissivity_31_1.jpg")
    return 0


def read_raw_data(emissivity_set):
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)
    emi_file = os.path.join(homefolder, 'hypothetical', 'emissivity_' + emissivity_set + '.txt')
    emi_raw = np.loadtxt(emi_file)
    return emi_raw


if 1:
    start_time = time.perf_counter()
    extrapolate_demo()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")