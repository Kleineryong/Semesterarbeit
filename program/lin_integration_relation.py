import os
import multiprocessing as mp
import numpy as np
import math
import warnings
import openpyxl
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')

import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from joblib import Parallel, delayed


def lin_integration_relation(emissivity_set, t_lb, t_ub):

    melt_temperature = 1600
    # generate temperature field
    t_field = np.array(range(t_lb, t_ub))

    dv_integ = dv_integration(t_field, emissivity_set, melt_temperature)
    dv_lin = dv_linear(t_field, emissivity_set, melt_temperature)
    ratio = dv_lin / dv_integ

    x = np.arange(ratio.shape[0])
    y = t_field # np.arange(ratio.shape[1])
    x, y = np.meshgrid(y, x)
    z = ratio
    ratio_max = np.max(ratio)
    ratio_min = np.min(ratio)

    dir_name = str(emissivity_set) + '_digital'
    result_dir = os.path.join('results', 'lin_inte_comp')
    if not os.path.exists(os.path.join(result_dir, dir_name)):
        os.mkdir(os.path.join(result_dir, dir_name))
    fig_1 = plt.figure()
    ax = fig_1.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=5, c='g', marker='1')

    ax.set_xlabel('temperature')
    ax.set_ylabel('channel')
    ax.set_zlabel('ratio')

    plt.savefig(os.path.join(result_dir, dir_name, 'ratio.jpg'))
    plt.clf()
    plt.close(fig_1)

    for i in range(ratio.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(t_field, ratio[i, :], s=15, c='g', marker='1')
        ax.set_ylim([ratio_min, ratio_max])

        # 设置坐标轴名称
        ax.set_xlabel('temperature')
        ax.set_ylabel('dv_lin / dv_integration')

        # 保存图形
        plt.savefig(os.path.join(result_dir, dir_name, 'channel_' + str(i) + '.jpg'))
        plt.clf()
        plt.close(fig)
    return 0


#########################################################
# physical value calculation
#########################################################
def dv_linear(t_field, emissivity_set, melt_temperature):
    # parameter setting
    wavelength = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.826]) * 10 ** (-6)
    sens_factor = np.array(
        [1754.7780081330168, 4614.964564504549, 9544.836689465254, 18681.526160164747, 28448.45940189293,
         42794.15859091206, 61722.9332334226, 89214.96448715679])
    sens_factor_b = [-625203.4109383911, -1255959.3399823632, -2280428.045299191, -2896486.193421305,
                     -3511055.365513118, -2647879.5561356354, -496042.6119804597, 6119684.353094454]
    exposure_time = 200e-3 * 10
    radiation = np.zeros([len(wavelength), len(t_field)])
    for i in range(len(wavelength)):
        for j in range(len(t_field)):
            radiation[i, j] = black_body_radiation(t_field[j], wavelength[i]) * emissivity(t_field[j], wavelength[i],
                                                                                           emissivity_read(emissivity_set), melt_temperature, emissivity_set)
    for i in range(len(t_field)):
        radiation[:, i] = (radiation[:, i] - sens_factor_b) * exposure_time / sens_factor
    return radiation


def dv_integration(t_field, emissivity_set, melt_temperature):
    # read camera parameter
    qe_address = os.path.join('camera_parameter', 'CMS22010236.xlsx')
    tr_address = os.path.join('camera_parameter', 'FIFO-Lens_tr.xls')
    cam_param = read_cam(qe_address, tr_address)
    cam_efficiency = cam_param['camera_total_efficiency']

    interp_emissivity = emissivity_read(emissivity_set)

    final_results_integration = {}
    # start calculation in channel
    for channel in cam_efficiency.keys():
        cam_efficiency_channel = [[int(k), v] for k, v in cam_efficiency[channel].items()]
        cam_efficiency_function = interp1d([i[0] for i in cam_efficiency_channel],
                                           [i[1] for i in cam_efficiency_channel], kind='linear',
                                           fill_value='extrapolate')

        def radiation_integration(temperature, emissivity_data, cam_efficiency_data):
            def radiation(wl):
                return black_body_radiation(temperature, wl) * emissivity(temperature, wl, emissivity_data,
                                                                          melt_temperature,
                                                                          emissivity_set) * camera_efficiency(wl,
                                                                                                              cam_efficiency_data)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                radiation_rec = quad(radiation, 500 * (10 ** -9), 900 * (10 ** -9), limit=5)[
                                    0] * 200  # 200 means the calibration factor and shutter time
            return radiation_rec

        digital_value = Parallel(n_jobs=mp.cpu_count() - 1)(
            delayed(radiation_integration)(t_field[i], interp_emissivity, cam_efficiency_function) for i in
            range(len(t_field)))
        final_results_integration[channel] = np.array(digital_value).astype(int)
        final_results_array = np.array([final_results_integration[key] for key in sorted(final_results_integration.keys())])
    return final_results_array[0:8, :]


def camera_efficiency(wavelength, cam_fun):
    qe = cam_fun(wavelength * (10**9))
    return qe


def read_cam(QE_address, T_address):
    QE_raw = pd.read_excel(QE_address, 'QE')
    T_raw = pd.read_excel(T_address)
    result = {}
    camera_total_efficiency = {}
    result['quantum_efficiency'] = QE_raw.to_numpy()[:, 1::]
    wavelength_set = QE_raw.to_numpy()[:, 0]
    t_interp = interp1d(T_raw.to_numpy()[:, 0]*1000, T_raw.to_numpy()[:, 1], kind='linear', fill_value='extrapolate')
    result['transparency'] = t_interp(wavelength_set)
    result['wavelength_set'] = wavelength_set
    wl, channel = np.shape(QE_raw.to_numpy()[:, 1::])
    for i in range(channel):
        camera_total_efficiency['channel_' + str(i)] = {}
        for j in range(wl):
            camera_total_efficiency['channel_' + str(i)][str(int(wavelength_set[j]))] = result['quantum_efficiency'][j, i] * result['transparency'][j]
    result['camera_total_efficiency'] = camera_total_efficiency
    return result


def emissivity(temperature, wavelength, emissivity_data, melt_temperature, emissivity_set):
    emissivity = emissivity_data(wavelength * 1e9) * factor_temperature(temperature, melt_temperature, emissivity_set)
    return min(emissivity, 1)


# read emissivity data
def emissivity_read(emissivity_set):
    emissivity_address = os.path.join("hypothetical", "emissivity_") + str(
        emissivity_set) + ".txt"  # relative address of emissivity file
    data_raw = np.loadtxt(emissivity_address)  # read raw data
    interp_emissivity = interp1d(data_raw[:, 0], data_raw[:, 1], kind='linear',
                                 fill_value='extrapolate')  # linear interpolation
    return interp_emissivity


# calculate temperature factor of emissivity
def factor_temperature(temperature, melt_temperature, emissivity_set):
    if emissivity_set:
        if temperature <= melt_temperature:
            factor = (1 - (temperature - 1500) / (2000 - 1500) * 0.2)
        else:
            factor = (1 - (temperature - 1500) / (2000 - 1500) * 0.2) # * 0.1
    else:
        if temperature <= melt_temperature:
            factor = 1
        else:
            factor = 0.1
    return factor


def black_body_radiation(temperature, wavelength):
    h = 6.62607015e-34      # Plank's constant
    c = 299792458           # Speed of light
    k_b = 1.380649e-23      # Boltzmann's constant
    radiance = 2 * h * np.float_power(c, 2) * np.float_power(wavelength, -5) / (math.exp(h*c/(k_b * wavelength * temperature))-1)
    return radiance


if 1:
    start_time = time.perf_counter()
    emissivity_set = ['21', '22', '23', '24', '25', '26', '31', '32', '33', '34']
    for i in emissivity_set:
        lin_integration_relation(i, 1500, 2000)
        print(i, '_finished')
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")