import os
import multiprocessing as mp
import numpy as np
import math
import warnings
import openpyxl
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import pandas as pd
from scipy.interpolate import interp1d


def emissivity():
    emissivity_set = [0, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34]

    #############################################
    # generate figure about raw emissivity data #
    # for i in emissivity_set:
    #     read_emissivity(str(i))
    #############################################

    #############################################
    # visualization of emissivity model
    emissivity_set = '33'
    melt_temperature = 1700
    emi_raw = read_raw_data(emissivity_set)
    emi_fun = interp1d(emi_raw[:, 0], emi_raw[:, 1], fill_value='extrapolate')

    t_field = np.linspace(1500, 2000, 30)
    wl_set = np.linspace(400, 800, 30)
    T, WL = np.meshgrid(t_field, wl_set)
    emi = emissivity_model(T, WL, emi_fun, melt_temperature, emissivity_set)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    # 绘制散点图
    ax.scatter(T, WL, emi, c='g')

    # 设置坐标轴标签
    ax.set_xlabel('Temperature[K]')
    ax.set_ylabel('Wavelength[nm]')
    ax.set_zlabel('Emissivity')

    plt.savefig('emissivity_model.jpg')
    return 0


def factor_temperature(temperature, melt_temperature, emissivity_set):
    if emissivity_set:
        if temperature <= melt_temperature:
            factor = (1 - (temperature - 1500) / (2000 - 1500) * 0.2)
        else:
            factor = (1 - (temperature - 1500) / (2000 - 1500) * 0.2) * 0.1
    else:
        if temperature <= melt_temperature:
            factor = 1
        else:
            factor = 0.1
    return factor


def emissivity_model(temperature, wavelength, emi_function, melt_temperature, emissivity_set):
    factor_vec = np.vectorize(factor_temperature)
    fact_temp = factor_vec(temperature, melt_temperature, emissivity_set)
    emi = emi_function(wavelength) * fact_temp
    return emi


def read_raw_data(emissivity_set):
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)
    emi_file = os.path.join(homefolder, 'hypothetical', 'emissivity_' + emissivity_set + '.txt')
    emi_raw = np.loadtxt(emi_file)
    return emi_raw


# generate figure about raw emissivity data
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