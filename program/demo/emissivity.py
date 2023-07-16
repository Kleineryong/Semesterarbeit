import os
import multiprocessing as mp
import numpy as np
import math
import warnings
import openpyxl
import time
import matplotlib.pyplot as plt

import pandas as pd
from scipy.interpolate import interp1d


def emissivity():
    emissivity_set = [0, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34]
    for i in emissivity_set:
        read_emissivity(str(i))
    return 0


def read_emissivity(emissivity_set):
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)
    emi_file = os.path.join(homefolder, 'hypothetical', 'emissivity_' + emissivity_set + '.txt')
    emi_raw = np.loadtxt(emi_file)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(emi_raw[:, 0], emi_raw[:, 1])
    plt.xlabel('wavelength[nm]', fontsize=30)
    plt.ylabel('emissivity', fontsize=30)
    plt.tick_params(axis='x', labelsize=15)  # 设置x轴刻度数字的大小为10
    plt.tick_params(axis='y', labelsize=15)  # 设置y轴刻度数字的大小为10
    plt.xlim(390, 830)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(homefolder, 'demo', 'emissivity_' + emissivity_set + '.jpg'))
    return 0


if 1:
    start_time = time.perf_counter()
    emissivity()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")