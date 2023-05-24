import math
import numpy as np
from scipy.interpolate import interp1d


#########################################################
# construct black body radiation field
# based on plank's law
# input: temperature[K], start_wavelength[nm], end_wavelength[nm], wavelength_interval[nm]
# output: radiation at each wavelength
#########################################################
def black_body_radiation(temperature, melt_temperature, start_wavelength, end_wavelength, wavelength_interval, type, data_set):
    radiation = {}
    for i in range(start_wavelength, end_wavelength, wavelength_interval):
        ###############
        # use one of the following emissivity model
        if type == 'model':
            radiation[str(i)] = emissivity_model(temperature, melt_temperature, i) * plank_law(temperature, i)        # use emissivity model
        elif type == 'data':
            radiation[str(i)] = emissivity_data(temperature, data_set, i) * plank_law(temperature, i)      # use emissivity_data
        else:
            print('wrong type')
        ###############
    return radiation


########################################################
# used for calculating the black body radiation        
########################################################
def plank_law(temperature, wavelength):
    h = 6.62607015e-34      # Plank's constant
    c = 299792458           # Speed of light
    k_b = 1.380649e-23      # Boltzmann's constant

    radiance = 2 * h * np.float_power(c, 2) * np.float_power(wavelength/1e9, -5) / (math.exp(h*c/(k_b * wavelength/1e9 * temperature))-1)

    return radiance


########################################################
# emissivity model
# used for generating emissivity of hypothetical material
# temperature range: 1500-2000[k]
# wavelength_range: 500-900[nm]
# input: temperature[K], melt_temperature[k], wavelength[nm]
# output: emissivity
# details of emissivity model can be seen in scratch.py
########################################################
def emissivity_model(temperature, melt_temperature, wavelength):
    # reference point:
    epsilon_1500k = {}
    start_wavelength = 500
    end_wavelength = 900

    epsilon_1500k[str(start_wavelength)] = 0.35
    epsilon_1500k[str(end_wavelength)] = 0.28

    emissivity_liquid = 0.1
    factor_scale = (epsilon_1500k[str(start_wavelength)] - emissivity_liquid)/epsilon_1500k[str(start_wavelength)]                      # used to correct the amplitude of sigmoid function

    k_lambda = (epsilon_1500k[str(end_wavelength)] - epsilon_1500k[str(start_wavelength)]) / (end_wavelength - start_wavelength)        # describe the k in 'emissivity = k * lambda + b'

    factor_temperature = (1/(1 + 1/math.exp((melt_temperature - temperature)/15))) * (1 - (temperature - 1500)/(2000-1500) * 0.2)       # 15 is the stiffness of phase change,
    emissivity = (k_lambda * (wavelength - start_wavelength) + epsilon_1500k[str(start_wavelength)]) * factor_temperature * factor_scale + emissivity_liquid

    return emissivity


#################################################
# emissivity data
# read emissivity data from file
# file format: .txt, include wavelength and the correspoding emissivity
# input: temperature, (melt_temperature), wavelength
#################################################
def emissivity_data(temperature, material_type, wavelength):
    file_address = "hypothetical\emissivity_" + str(material_type) + ".txt"      # relative address of emissivity file
    data_raw = np.loadtxt(file_address)                 # read raw data
    interp_emissivity = interp1d(data_raw[:, 0], data_raw[:, 1], kind = 'linear', fill_value='extrapolate')       # linear interpolation
    emissivity = interp_emissivity(int(wavelength)) * 0.2 * (temperature - 1500) / 500 + interp_emissivity(int(wavelength))
    return emissivity
