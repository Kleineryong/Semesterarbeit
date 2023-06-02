import os
import multiprocessing as mp
import numpy as np
import math
import warnings
import openpyxl
import time
from scipy.constants import c, h, k
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize

import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from joblib import Parallel, delayed


##############################################################
# main program
# dimension: wavelength[nm]
##############################################################
def t_estimate_integration():
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)
    result_dir = 'result_v010_lin'
    result_dir = os.path.join('results', result_dir)
    data_temperature = '1896'
    emissivity_set = '2'

    # camera parameter reading
    camera_folder = os.path.join(homefolder, 'program', 'camera_parameter')
    transparency, qe_total = read_cam(camera_folder)

    # intensity data reading
    data_name = 'T' + data_temperature + '_' + emissivity_set + '_digital'
    intensity_dir = os.path.join(homefolder, 'program', 'data', data_name)
    intensity_raw = read_intensity(intensity_dir, data_temperature) # used to control the calculate range
    intensity_target = intensity_raw[:, 23:27, 23:27]

    # transform raw data to enable parallel calculation
    intensity_reshape = transfer_pos(intensity_target)
    print("Number of processors: ", mp.cpu_count())
    # cal_result = np.array(Parallel(n_jobs=mp.cpu_count())(
    #     delayed(process_itg)(intensity_reshape[i], qe_total, transparency) for i in range(len(intensity_target[0]))))
    temp = process_itg(intensity_reshape[0], qe_total, transparency)
    # test2 = transfer_neg(test1[:, 0], intensity_target)
    return 0


##############################################################
# functional unit
##############################################################
def radiation(wavelength, transparency, qe_array, emissivity_param, t):
    qe = interp1d(qe_array[0], qe_array[1])
    result = transparency(wavelength) * emissivity_model(wavelength, emissivity_param) \
             * black_body_radiation(t, wavelength) * qe(wavelength) * 200
    return result


def emissivity_model(wavelength, *param):
    wl0 = 0.5 * 10 ** (-6)
    wl1 = 1 * 10 ** (-6)
    wl_rel = (wavelength-wl0)/(wl1-wl0)
    # lin emi = a + b * lambda
    emissivity = param[0] - param[1] * wl_rel   # a[0, 1], b[0, 1]

    # lin exp emi = exp(a + b * lambda)
    # emissivity = math.exp(param[0] + param[1] * wl_rel)

    # lin square_2 emi = a + b * wl**2
    # emissivity = param[0] + param[1] * (wl_rel**2)

    # lin square_2 emi = a + b * wl + c * wl**2
    # emissivity = param[0] + param[1] * wl_rel + param[2] * (wl_rel ** 2)

    # exp emi = exp(-a - b * wl)
    # emissivity = math.exp(-param[0] - param[1] * wl_rel)

    # emissivity = param

    return emissivity


def process_itg(intensity, quantum_efficiency, transparency):
    wl0 = 500e-9
    wl1 = 1000e-9
    def camera_model(qe, *emissivity_param, t):
        result_f = []
        for i in range(8):
            funct = quad(radiation, wl0, wl1, args=(transparency, qe[i], emissivity_param, t), epsabs=1e-2, limit=5)[0]
            result_f.append[funct]
        return np.array(result_f)
    bounds_t = (500, 2000)
    bounds_param = ([0, 0], [1, 1])
    bounds = [bounds_param, bounds_t]

    param_p0 = [0.5, 0.5]
    t_p0 = 1000
    initial_guess = [*param_p0, t_p0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        popt, cov = curve_fit(camera_model, quantum_efficiency, intensity, bounds=bounds, p0 = initial_guess, maxfev=100000)
    return popt

def transfer_pos(raw_data):
    return raw_data.reshape(raw_data.shape[0], -1).T


def transfer_neg(raw_data, target_value):
    shape_data = [target_value.shape[1], target_value.shape[2]]
    return raw_data.reshape(shape_data)


def read_intensity(intensity_dir, data_temperature):
    intensity = [
        pd.read_excel(os.path.join(intensity_dir, f"digital_value_{data_temperature}.xlsx"), 'channel_' + str(i),
                      header=None)
        for i in range(8)]
    intensity = np.array(intensity)
    return intensity


def read_cam(camera_dir):
    result = {}
    df_qe = pd.read_excel(os.path.join(camera_dir, "CMS22010236.xlsx"), 'QE')
    df_tr = np.array(pd.read_excel(os.path.join(camera_dir, "FIFO-Lens_tr.xls"))).transpose()

    # transparency
    transparency = interp1d((df_tr[0] * 1e-6), df_tr[1], kind='linear', fill_value='extrapolate')

    # quantum efficiency
    qe_array = np.array(df_qe.iloc[:, 0:9]).T
    qe_array[0] *= 1e-9
    qe = np.zeros([qe_array.shape[0]-1, 2, qe_array.shape[1]])
    qe[:, 0, :] = qe_array[0]
    for i in range(8):
        qe[i, 1, :] = qe_array[i+1]
    return transparency, qe


def black_body_radiation(temperature, wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    return param1 / (wavelength**5) / (np.exp(param2 / (wavelength * temperature)) - 1)


if 1:
    start_time = time.perf_counter()
    t_estimate_integration()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")