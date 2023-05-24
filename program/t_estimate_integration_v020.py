import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from scipy.constants import c, h, k
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad

def t_estimate_integration():
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)

    data_temperature = '1897'
    emissivity_set = '0'
    data_name = 'T' + data_temperature + '_' + emissivity_set + '_digital'

    # wl = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.82]) * 10 ** (-6)
    camera_folder = os.path.join(homefolder, 'program', 'camera_parameter')
    # exp_case = ["_e005", "_e01", "_e02", "_e03", "_e05", "_e1"]
    DF_QE = pd.read_excel(os.path.join(camera_folder, "CMS22010236.xlsx"), 'QE')
    DF_T = pd.read_excel(os.path.join(camera_folder, "FIFO-Lens_tr.xls"))


    tr_array = np.array(DF_T).transpose()
    qe_array = []
    for i in range(8):
        qe_array.append(DF_QE.iloc[:, [0, 1+i]])
    qe_array = np.array(qe_array).transpose(0, 2, 1)
    # print(qe_array)
    experiment_folder = os.path.join(homefolder, 'program', 'data', data_name)
    # print(experiment_folder)
    intensity = []
    for i in range(8):
        intensity.append(pd.read_excel(os.path.join(experiment_folder, ('digital_value_' + data_temperature + '.xlsx')), 'channel_' + str(i), header=None))
    t_ref = np.array(pd.read_excel(os.path.join(experiment_folder, 't_field_' + data_temperature + '.xlsx'), header=None))
    intensity = np.array(intensity)
    target = intensity #[:, 15:25, 15:25]
    t_map = np.zeros((len(target[0]), len(target[0, 0])))
    Ea_map = np.zeros((len(target[0]), len(target[0, 0])))
    Eb_map = np.zeros((len(target[0]), len(target[0, 0])))
    print(target)

    for i in range(len(target[0])):
        for j in range(len(target[0, 0])):
            popt = process_itg(target[:, i, j], qe_array, tr_array)
            t_map[i, j] = popt[0]
            Ea_map[i, j] = popt[1]
            Eb_map[i, j] = popt[2]

    save_dir = os.path.join(homefolder, 'program', 'result_v020', 'T' + data_temperature + '_' + emissivity_set + '_digital')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df1 = pd.DataFrame(t_map)
    df1.to_excel(save_dir + "/" + 'T.xlsx', index=False)
    df2 = pd.DataFrame(Ea_map)
    df2.to_excel(save_dir + "/" + 'Ea.xlsx', index=False)
    df3 = pd.DataFrame(Eb_map)
    df3.to_excel(save_dir + "/" + 'Eb.xlsx', index=False)
    df4 = pd.DataFrame(t_map - t_ref)#[15:25, 15:25])
    df4.to_excel(save_dir + "/" + 'T_ab.xlsx', index=False)


# def normalize_data(data):
#     min_value = min(data)
#     max_value = max(data)
#     normalized_data = [(x - min_value)/(max_value - min_value) for x in data]
#     return normalized_data


def lin_interpolation(x, x0, x1, y0, y1):
    return y0+(y1-y0)*(x-x0)/(x1-x0)


def black_body_radiation(temperature, wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    return param1/(wavelength**5)/(np.exp(param2/(wavelength*temperature))-1)


# def integration(wl, tr_array, qe_array, a, b, t):
#     wl0 = 0.5 * 10 ** (-6)
#     wl1 = 1 * 10 ** (-6)
#     tr_i = len(tr_array[0, tr_array[0, :] * 10**(-9) <= wl]) - 1
#     qe_i = len(qe_array[0, qe_array[0, :] * 10**(-9) <= wl]) - 1
#     # print(len(tr_array[0, tr_array[0, :] * 10**(-9) <= wl]), wl,tr_i)
#     tr = lin_interpolation(wl * 10 ** 9, tr_array[0, tr_i - 1], tr_array[0, tr_i], tr_array[1, tr_i - 1],
#                            tr_array[1, tr_i])
#     qe = lin_interpolation(wl * 10 ** 9, qe_array[0, qe_i - 1], qe_array[0, tr_i], qe_array[1, qe_i - 1],
#                            qe_array[1, qe_i])
#     return tr * (a - b * (wl - wl0) / (wl1 - wl0)) * qe * black_body_radiation(t, wl) * 200

def integration(wl,f_array,qe_array,a,b,t):
    wl0 = 0.5 * 10 ** (-6)
    wl1 = 1 * 10 ** (-6)
    f_i = len(f_array[0,f_array[0,:]*10**(-9)<=wl]) - 1
    qe_i = len(qe_array[0,qe_array[0,:]*10**(-9)<=wl]) - 1
    f = lin_interpolation(wl*10**9,f_array[0,f_i-1],f_array[0,f_i],f_array[1,f_i-1],f_array[1,f_i])
    qe = lin_interpolation(wl*10**9,qe_array[0,qe_i-1],qe_array[0,qe_i],qe_array[1,qe_i-1],qe_array[1,qe_i])
    result = f*(a-b*(wl-wl0)/(wl1-wl0))*qe*black_body_radiation(t,wl)*200
    return result


def process_itg(intensity_array, qe_array, tr_array):
    wl0 = 0.5 * 10 ** (-6)
    wl1 = 1 * 10 ** (-6)
    def integration_solve(qe, a, b, t):
        result_f = []
        for i in range(8):
            funct = quad(integration, wl0, wl1, args=(tr_array, qe[i], a, b, t), epsabs = 1e-2, limit = 5)[0]
            result_f.append(funct)
        return np.array(result_f)
    popt, cov = curve_fit(integration_solve, qe_array, intensity_array, bounds = ((0, 0, 500), (1, 1, 1958.2)), maxfev= 100000)
    return popt[2], popt[0], popt[1]


if 1:
    start_time = time.perf_counter()
    t_estimate_integration()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")