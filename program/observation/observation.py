import numpy as np
import math
import matplotlib.pyplot as plt
import tifffile
import os
import xlwt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from joblib import Parallel, delayed

from hypothetical import hypothetical


########################################
# radiation model
# used to describe the observation in different channel
# input: wavelength_set, emissivity_type["model" / "data"], emissivity_set, melt_temperature[k], resolution, diameter_ratio, temperature_center[k], temperature_background[k]
# output:output['radiation'], output['temperature']
########################################
def radiation_model(wavelength_set, emissivity_type, emissivity_data_set, melt_temperature, resolution,
                      diameter_ratio, temperature_center, temperature_background, distribution_type):
    t_field = temperature_field(resolution, diameter_ratio, temperature_center, temperature_background,
                                distribution_type)
    channel = {}
    output = {}
    plank_law_vec = np.vectorize(hypothetical.plank_law)
    emissivity_model_vec = np.vectorize(hypothetical.emissivity_model)
    emissivity_data_vec = np.vectorize(hypothetical.emissivity_data)
    emi_workbook = xlwt.Workbook(encoding='utf-8')          # save to xls
    for wavelength in wavelength_set:
        sheet = emi_workbook.add_sheet(str(wavelength))     # save to xls
        if emissivity_type == 'model':
            emissivity = emissivity_model_vec(t_field, melt_temperature, float(wavelength))
        elif emissivity_type == 'data':
            emissivity = emissivity_data_vec(t_field, emissivity_data_set, float(wavelength))
        channel[str(wavelength)] = plank_law_vec(t_field, float(wavelength)) * emissivity
        save_radiation(channel[str(wavelength)], temperature_center, str(wavelength), emissivity_type,
                       emissivity_data_set)
        for i in range(len(emissivity)):                    # save to xls
            for j in range(len(emissivity[i])):             # save to xls
                sheet.write(i, j, emissivity[i][j])         # save to xls
    save_add_info(t_field, emi_workbook, temperature_center, emissivity_type, emissivity_data_set)
    output['temperature'] = t_field
    output['radiation'] = channel
    output['emissivity'] = emissivity

    return output


########################################
# real radiation model without saving
# used to calculate real radiation in the field in corresponding wavelength
# input: t_field[k], wavelength_set, emissivity_type["data"], emissivity_set, melt_temperature[k], temperature_center[k], temperature_background[k]
# output:output['radiation'], output['temperature']
########################################
def radiation_model_conti(t_field, wavelength_set, emissivity_type, emissivity_data_set, melt_temperature, temperature_center):
    channel = {}
    output = {}
    plank_law_vec = np.vectorize(hypothetical.plank_law)
    # emissivity_model_vec = np.vectorize(hypothetical.emissivity_model)
    emissivity_data_vec = np.vectorize(hypothetical.emissivity_data)

    for wavelength in wavelength_set:
        # if emissivity_type == 'model':
        #     emissivity = emissivity_model_vec(t_field, mel t_temperature, float(wavelength))
        # elif emissivity_type == 'data':
        #     emissivity = emissivity_data_vec(t_field, emissivity_data_set, float(wavelength))
        emissivity = emissivity_data_vec(t_field, emissivity_data_set, wavelength)
        channel[str(int(wavelength))] = plank_law_vec(t_field, wavelength) * emissivity

    output['radiation'] = channel
    return output


########################################
# camera model
# transfer real radiation data to digital value
# input: real_radiation[dict], camera_efficiency[dict], explosure_time[s]
# output: digital_value
########################################
def camera_model(real_radiation, camera_efficiency, shutter_time):
    digital_value = {}
    # transfer the dictionary data to ndarray
    radiation = dict_array(real_radiation)

    # print(radiation[:, :, 1])
    for channel in sorted(camera_efficiency.keys()):
        digital_value[channel] = eff_radiation(radiation, camera_efficiency[channel], shutter_time)
    return digital_value


########################################
# dict_array
# transfer real_radiation into 3D array
# input: real_radiation
# output: radiation_array
########################################
def dict_array(real_radiation):
    wavelength_size = len(real_radiation.keys())
    radiation_array = np.zeros([real_radiation[list(real_radiation.keys())[0]].shape[0],
                                real_radiation[list(real_radiation.keys())[0]].shape[1],
                                wavelength_size])
    for wl in range(wavelength_size):
        radiation_array[:, :, wl] = real_radiation[sorted(real_radiation.keys(), key=lambda x: int(x))[wl]]
    return radiation_array


########################################
# effective radiation
# calculate effective radiation value in single point
# input: real_radiation_point[ndarray], camera_eff[dict, 'wl':float]
# output: eff_radiation[float]
########################################
def eff_radiation(real_radiation, camera_efficiency, shutter_time):
    wavelength_set = sorted(camera_efficiency.keys(), key=lambda x: int(x))
    # used for parallel computing
    wavelength_set_int_nm = list(map(int, wavelength_set))
    wavelength_set_int = [num * 1e-9 for num in wavelength_set_int_nm]


    qe_set = np.zeros(len(wavelength_set))
    for i in range(len(wavelength_set)):
        qe_set[i] = camera_efficiency[wavelength_set[i]]

    # reshape the real_radiation field and multiply with the camera_efficiency
    real_radiation_reshape = real_radiation.reshape(real_radiation.shape[0] * real_radiation.shape[1], real_radiation.shape[2]) * qe_set
    digital_value = Parallel(n_jobs=-1)(delayed(integration_radiation)(wavelength_set_int, real_radiation_reshape[i, :], shutter_time) for i in range(len(real_radiation_reshape)))
    # print(real_radiation_reshape.shape)
    result = np.array(digital_value).reshape([real_radiation.shape[0], real_radiation.shape[1]])
    # print(result.shape)
    return result


########################################
# calculation the intergration of a single point
# input: wavelength[list], radiation[list]
# output: digital_value[double]
########################################
def integration_radiation(wl, radi, shutter_time):
    fun = interp1d(wl, radi)
    result = quad(fun, wl[0], wl[-1])
    return result[0] * shutter_time


########################################
# temperature_field
# used to generate the temperature field
# input: resolution[list], diameter_ratio[], temperature_center[K], temperature_background[K], distribution_type[string]
# output: temperature_field[np.array]
########################################
def temperature_field(resolution, diameter_ratio, temperature_center, temperature_background, distribution_type):
    # calculation of geometric relations
    row_center = resolution[0] / 2
    colum_center = resolution[1] / 2
    radius = round(min(resolution[0], resolution[1]) * diameter_ratio / 2)
    sigma = radius * 2.5

    # initialise the temperature field
    field = np.ones(resolution)

    # start calculation
    if distribution_type == 'sigmoid':
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                field[i, j] = round((temperature_background + (temperature_center - temperature_background) *
                                    sigmoid(-((i - row_center)**2 + (j - colum_center)**2) + radius**2)) *
                                    distribution(((i - row_center)**2 + (j - colum_center)**2), sigma))
    elif distribution_type == 'linear':
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                field[i, j] = max(round(temperature_center + (temperature_background - temperature_center) *
                                        np.sqrt((i - row_center)**2 + (j - colum_center)**2) / radius), temperature_background)
    elif distribution_type == 'gaussian':
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                field[i, j] = max(round(temperature_background + (temperature_center - temperature_background) *
                                        distribution(((i - row_center)**2 + (j - colum_center)**2), sigma)),
                                  temperature_background)
    return field


########################################
# noise model
# used to add noise signal in the measurement
# input:
# output:
########################################
def noise_model():
    output = 1
    return output


########################################
# sigmoid function
# used to calculate sigmoid function
########################################
def sigmoid(x):
    return 1. / (1. + math.exp(-x/100))


########################################
# distribution function
# used to calculate the temperature distribution
########################################
def distribution(x, sigma):
    result = 1 + 1 / (math.sqrt(2 * math.pi)* sigma) / math.exp(x / (2 * sigma))*50
    return result


########################################
# plot a field in 3d vision
########################################
def temperature_plot(field):
    row = []
    colum = []
    value = []
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            row.append(i)
            colum.append(j)
            value.append(field[i, j])

    # Creating figure
    fig = plt.figure(figsize=(15, 11))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(row, colum, value, color="green")
    plt.title("temperature")
    plt.xlabel('x_coordinate')
    plt.ylabel('y_coordinate')
    ax.set_zlabel('temperature[k]')

    # show plot
    plt.show()
    return 0


##########################################
# save file in data
##########################################
def save_radiation(rad_field, temperature, wavelength, emissivity_type, emissivity_data_set):
    dir_name = 'T' + str(temperature) + '_' + emissivity_type + str(emissivity_data_set)
    if not os.path.exists(os.path.join('data', dir_name)):
        os.mkdir(os.path.join('data', dir_name))
    file_name = 'T' + str(temperature) + '_channel_' + str(wavelength) + '.tiff'
    tifffile.imwrite(os.path.join('data',dir_name,file_name), rad_field)

    field_norm = (norm_cal(rad_field))
    dir_name_norm = dir_name + '_norm'
    if not os.path.exists(os.path.join('data', dir_name_norm)):
        os.mkdir(os.path.join('data', dir_name_norm))
    tifffile.imwrite(os.path.join('data', dir_name_norm, file_name), field_norm.astype(int))
    return 0


def save_add_info(t_field, emi_field, temperature, emissivity_type, emissivity_data_set):
    dir_name = 'T' + str(temperature) + '_' + emissivity_type + str(emissivity_data_set) + '_add_info'
    if not os.path.exists(os.path.join('data', dir_name)):
        os.mkdir(os.path.join('data', dir_name))
    emi_field.save(os.path.join('data', dir_name, 'emi_field.xls'))
    np.savetxt(os.path.join('data', dir_name, 't_field.txt'), t_field)
    return 0


##########################################
# normalisation
# transfer physic value into digital value (simulate the sensor)
# rule: set radiation at 2000K, 500nm to 1024,
# input: field
# output: 0
##########################################
def norm_cal(field):
    # linear-relation
    max = hypothetical.plank_law(1300, 900)
    min = 0
    max_value = 1023
    field_new = field / (max-min) * max_value
    return field_new