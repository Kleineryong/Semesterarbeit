import os
import multiprocessing as mp
import numpy as np
import math
import warnings
import openpyxl
import time
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
    # set save parameters
    temperature_center = 1300
    emissivity_set = 3
    # set intensity data address
    intensity_digital_address = os.path.join('data', 'T1300_3_digital', 'digital_value_1300.xlsx')
    intensity_raw_data = intensity_reshape(read_digital_value(intensity_digital_address))
    intensity_digital_dict = intensity_raw_data['intensity_dict']
    intensity_digital_array = intensity_raw_data['intensity_array']
    # calculate dimension of raw data
    intensity_digital_shape = intensity_raw_data['shape']

    # print(intensity_digital['channel_0'])
    print(intensity_digital_array[1])



    # set camera file address
    qe_address = os.path.join('camera_parameter', 'CMS22010236.xlsx')
    tr_address = os.path.join('camera_parameter', 'FIFO-Lens_tr.xls')

    # calculate camera efficiency
    cam_param = read_cam(qe_address, tr_address)
    cam_efficiency = cam_param['camera_function']

    shutter_time = 200
    # print(radiation_cal(600*(10**(-9)), cam_efficiency['channel_1'], 1, 0.01, 1500))
    # print(camera_model(cam_efficiency, 1, 0.01, 1500))

    # print(difference_cal([1, 1, 1500], cam_efficiency, [-127662.40620317, -66300.46491211, -38079.56121666, -12873.14080551, -17544.34037741, -12896.67251452, -17054.00386317, -19213.3940165, -218278.61885477]))
    # start minimization
    boundary = [(0, 1), (-0.1, 0.1), (500, 2000)]
    temperature_cal = np.empty(len(intensity_digital_array))
    emissivity_cal = np.empty(len(intensity_digital_array))
    for i in range(len(intensity_digital_array)):
        opt_result = minimize(difference_cal, [0.5, 0, 600], args=(cam_efficiency, intensity_digital_array[i]), bounds=boundary)
        temperature_cal[i] = opt_result.x[2]
        emissivity_cal[i] = emissivity_average_cal(opt_result.x[0], opt_result.x[1])
        print('a:', opt_result.x[0])
        print('b:', opt_result.x[1])
    temperature_cal = temperature_cal.reshape(intensity_digital_shape)
    emissivity_cal = emissivity_cal.reshape(intensity_digital_shape)
    save_file(temperature_cal, temperature_center, emissivity_set, emissivity_cal)
    return 0


##############################################################
# functional unit
##############################################################
def emissivity_average_cal(a, b):
    wl0 = 500
    wl1 = 901
    wavelength = np.arange(wl0, wl1)
    emissivity = np.average([(a + b * ((wl - wl1) / (wl0 - wl1))) for wl in wavelength])
    return emissivity

def save_file(t_field, temperature_center, emissivity_set, emi_field):
    dir_name = 'T' + str(temperature_center) + '_' + str(emissivity_set) + '_digital'
    if not os.path.exists(os.path.join('results', dir_name)):
        os.mkdir(os.path.join('results', dir_name))

    # t_field
    file_t = os.path.join('results', dir_name, 't_cal_' + str(temperature_center) + '.xlsx')
    workbook_t = openpyxl.Workbook()
    worksheet_t = workbook_t.active
    # image_t = Image.fromarray(t_field).convert("L")
    # image_t.save(os.path.join('data', dir_name, 't_field' + '.png'))
    plt.imshow(t_field, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Temperature_map')
    plt.savefig(os.path.join('results', dir_name, 't_cal' + '.jpg'))
    plt.clf()

    for row in t_field:
        worksheet_t.append(list(row))
    workbook_t.save(file_t)

    # emi_field
    file_emi = os.path.join('results', dir_name, 'emi_cal_' + str(temperature_center) + '.xlsx')
    workbook_emi = openpyxl.Workbook()
    worksheet_emi = workbook_emi.active
    # image_emi = Image.fromarray(emi_field).convert("L")
    # image_emi.save(os.path.join('data', dir_name, 'emi_field' + '.png'))
    plt.imshow(emi_field, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Emissivity_map')
    plt.savefig(os.path.join('results', dir_name, 'emi_cal' + '.jpg'))
    plt.clf()

    for row in emi_field:
        worksheet_emi.append(list(row))
    workbook_emi.save(file_emi)


def difference_cal(param, camera_efficiency, intensity_digital):
    a, b, temperature = param
    # call camera_model function with a, b, temperature as inputs
    intensity_cal = camera_model(camera_efficiency, a, b, temperature)
    # calculate the difference between the output and the intensity_digital array
    difference = intensity_cal - intensity_digital
    # calculate the sum of squares of the difference array
    sum_of_squares = (difference**2).sum()
    return sum_of_squares


def camera_model(camera_efficiency, a, b, temperature):
    wl_0 = 500 * 10**(-9)
    wl_1 = 1000 * 10**(-9)
    digital_value = []
    for channel in sorted(camera_efficiency.keys()):
        digital_value.append(quad(radiation_cal, wl_0, wl_1, args=(camera_efficiency[channel], a, b, temperature), epsabs = 1e-2, limit=10)[0])
    return np.array(digital_value)


######################
# calculate the radiation at a wavelength and temperature
# a, b is used to calculate emissivity. here we assum a linear relationship
# 200 is the shutter time
######################
def radiation_cal(wavelength, camera_function, a, b, temperature):
    # linear character of emissivity
    wl_0 = 500 * 10**(-9)
    wl_1 = 900 * 10**(-9)
    emissivity = a + b * (wavelength - wl_1) / (wl_0 - wl_1)
    radiation = camera_function(wavelength) * black_body_radiation(temperature, wavelength) * emissivity * 200
    return radiation


def intensity_reshape(intensity_raw):
    shape_raw_intensity = intensity_raw[list(intensity_raw.keys())[0]].shape
    intensity_1d = {}
    for channel in intensity_raw.keys():
        intensity_1d[channel] = intensity_raw[channel].reshape(shape_raw_intensity[0]*shape_raw_intensity[1])
    result = {}
    result['shape'] = shape_raw_intensity
    result['intensity_dict'] = intensity_1d
    # create empty array for reforming the intensity data to array
    intensity_array = np.empty((shape_raw_intensity[0] * shape_raw_intensity[1], len(intensity_1d.keys())))
    for i, key in enumerate(intensity_1d.keys()):
        intensity_array[:, i] = intensity_1d[key]
    result['intensity_array'] = intensity_array
    return result


def read_digital_value(intensity_address):
    data = pd.read_excel(intensity_address, sheet_name=None, header=None)
    digital_value = {}
    for sheet_name in data.keys():
        digital_value[sheet_name] = np.array(data[sheet_name])
    return digital_value


def read_cam(qe_address, tr_address):
    qe_raw = pd.read_excel(qe_address, 'QE')
    t_raw = pd.read_excel(tr_address)
    result = {}
    camera_total_efficiency = {}
    result['quantum_efficiency'] = qe_raw.to_numpy()[:, 1::]
    wavelength_set = qe_raw.to_numpy()[:, 0]
    t_interp = interp1d(t_raw.to_numpy()[:, 0]*1000, t_raw.to_numpy()[:, 1], kind='linear', fill_value='extrapolate')
    result['transparency'] = t_interp(wavelength_set)
    result['wavelength_set'] = wavelength_set
    wl, channel = np.shape(qe_raw.to_numpy()[:, 1::])
    # set camera efficiency into dict type
    for i in range(channel):
        camera_total_efficiency['channel_' + str(i)] = {}
        for j in range(wl):
            camera_total_efficiency['channel_' + str(i)][str(int(wavelength_set[j]))] = result['quantum_efficiency'][j, i] * result['transparency'][j]
    result['camera_total_efficiency'] = camera_total_efficiency
    # set camera efficiency into interp1d type
    camera_function = {}
    for channel in camera_total_efficiency.keys():
        camera_total_efficiency_array = [[int(k), v] for k, v in camera_total_efficiency[channel].items()]
        camera_efficiency_function = interp1d([i[0] for i in camera_total_efficiency_array],
                                           [i[1] for i in camera_total_efficiency_array], kind='linear',
                                           fill_value='extrapolate')
        camera_function[channel] = camera_efficiency_function
    result['camera_function'] = camera_function
    return result


def black_body_radiation(temperature, wavelength):
    h = 6.62607015e-34      # Plank's constant
    c = 299792458           # Speed of light
    k_b = 1.380649e-23      # Boltzmann's constant
    radiance = 2 * h * np.float_power(c, 2) * np.float_power(wavelength, -5) / (math.exp(h*c/(k_b * wavelength * temperature))-1)
    return radiance


if 1:
    start_time = time.perf_counter()
    t_estimate_integration()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")