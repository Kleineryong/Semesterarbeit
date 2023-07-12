# 适用于lin方法直接在真实实验数据中计算.
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
from PIL import Image


def t_estimate_v052():
    t_should = 700
    result_dir = 'result_v052_lin'

    data_dir = os.path.join('data', 'black_body_radiation', 'V0_bb_' + str(t_should))
    intensity_raw = read_tiff_images(data_dir)

    target = intensity_raw
    t_map = np.zeros((len(target[0]), len(target[0, 0])))
    Ea_map = np.zeros((len(target[0]), len(target[0, 0])))
    Eb_map = np.zeros((len(target[0]), len(target[0, 0])))
    emi_map = np.zeros((len(target[0]), len(target[0, 0])))

    # start calculation
    # parallel computing
    target_reshape = transfer_pos(target)
    print("Number of processors: ", mp.cpu_count())
    cal_result = np.array(Parallel(n_jobs=mp.cpu_count()-1)(
        delayed(process_itg)(target_reshape[:, i]) for i in range(len(target_reshape[0]))))
    t_map = transfer_neg(cal_result[:, 0], target)
    Ea_map = transfer_neg(cal_result[:, 1], target)
    Eb_map = transfer_neg(cal_result[:, 2], target)

    for i in range(Ea_map.shape[0]):
        for j in range(Ea_map.shape[1]):
            emi_map[i, j] = emissivity_average_cal(Ea_map[i, j], Eb_map[i, j])

    save_file(t_map, t_should, 0, emi_map, os.path.join('results', result_dir))

    return 0


def save_file(t_field, temperature_center, emissivity_set, emi_field, result_dir):
    dir_name = 'T' + str(temperature_center) + '_' + str(emissivity_set) + '_digital'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(os.path.join(result_dir, dir_name)):
        os.mkdir(os.path.join(result_dir, dir_name))

    # t_field
    file_t = os.path.join(result_dir, dir_name, 't_cal_' + str(temperature_center) + '.xlsx')
    workbook_t = openpyxl.Workbook()
    worksheet_t = workbook_t.active
    plt.imshow(t_field, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Temperature_map')
    plt.savefig(os.path.join(result_dir, dir_name, 't_cal' + '.jpg'))
    plt.clf()

    for row in t_field:
        worksheet_t.append(list(row))
    workbook_t.save(file_t)

    # emi_field
    file_emi = os.path.join(result_dir, dir_name, 'emi_cal_' + str(temperature_center) + '.xlsx')
    workbook_emi = openpyxl.Workbook()
    worksheet_emi = workbook_emi.active
    plt.imshow(emi_field, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Emissivity_map')
    plt.savefig(os.path.join(result_dir, dir_name, 'emi_cal' + '.jpg'))
    plt.clf()

    for row in emi_field:
        worksheet_emi.append(list(row))
    workbook_emi.save(file_emi)


def emissivity_average_cal(a, b):
    wl0 = 500
    wl1 = 1000
    wavelength = np.arange(wl0, wl1) * 10**(-9)
    emissivity = np.average([emissivity_model(wl, a, b) for wl in wavelength])
    return emissivity


def black_body_radiation(temperature, wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    return param1/(wavelength**5)/(np.exp(param2/(wavelength*temperature))-1)


def radiation_pv(wl, a, b, t):
    result = []
    for i in wl:
        result.append(emissivity_model(i, a, b) * black_body_radiation(t, i))
    return result


def emissivity_model(wl, a, b):
    wl0 = 0.5 * 10 ** (-6)
    wl1 = 0.9 * 10 ** (-6)
    wl_rel = (wl-wl0)/(wl1-wl0)

    # lin emi = a + b * lambda
    emissivity = a + b * wl_rel   # a[0, 1], b[-1, 1]

    # lin exp emi = exp(a + b * lambda)
    # emissivity = math.exp(a + b * wl)

    # lin square emi = a + b * wl**2
    # emissivity = a + b * (wl_rel**2)

    # exp emi = exp(-a - b * wl)
    # emissivity = math.exp(a - b * wl_rel)

    return max(0, min(1, emissivity))


def process_itg(intensity_array):
    wl0 = 0.5 * 10 ** (-6)
    wl1 = 1 * 10 ** (-6)
    exposure_time = 350e-4 #100e-4
    wl = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.826]) * 10 ** (-6)
    sens_factor = np.array(
        [1754.7780081330168, 4614.964564504549, 9544.836689465254, 18681.526160164747, 28448.45940189293,
         42794.15859091206, 61722.9332334226, 89214.96448715679])
    sens_factor_b = [-625203.4109383911, -1255959.3399823632, -2280428.045299191, -2896486.193421305,
                     -3511055.365513118, -2647879.5561356354, -496042.6119804597, 6119684.353094454]

    intensity_dv = intensity_array * sens_factor / exposure_time + sens_factor_b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, cov = curve_fit(radiation_pv, wl, intensity_dv, bounds=((0, -1, 500), (1, 1, 2000)), maxfev=100000)
    return popt[2], popt[0], popt[1]


def read_tiff_images(folder_path):
    image_list = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".tiff"):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            image_array = np.array(image)
            image_list.append(image_array)
    return np.array(image_list)


def transfer_pos(input):
    return input.reshape(input.shape[0], -1)


def transfer_neg(input, target_value):
    shape_data = [target_value.shape[1], target_value.shape[2]]
    return input.reshape(shape_data)


if 1:
    start_time = time.perf_counter()
    t_estimate_v052()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")