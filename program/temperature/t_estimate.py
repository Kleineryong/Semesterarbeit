import numpy as np
from scipy.optimize import minimize
from hypothetical import hypothetical
import math


######################################
# used to estimate the temperature based on input spectrum.
# input: wavelength[list], intensity_measurement[list]
# output: temperature, emissivity
######################################
def t_estimate(wavelength, intensity_meas):
    wavelength = np.array(wavelength, dtype='float')
    intensity_meas = np.array(intensity_meas)
    plank_law_vec = np.vectorize(hypothetical.plank_law)

    # set parameter
    par = np.array([0, 0, 0])       #par[0]: A; par[1]: B; par[2]: T_estimate;

    # target function. minimization of cost function
    h = 6.62607015e-34  # Plank's constant
    c = 299792458  # Speed of light
    k_b = 1.380649e-23  # Boltzmann's constant

    target_function = lambda par: np.sum(np.float_power(intensity_meas - (par[1] - par[0] * ((wavelength - 500) / (900 - 500))) * plank_law_vec(par[2]*np.ones(wavelength.size), wavelength), 2))

    # constraints 0 <= A < B <= 1
    # cons = ({'type': 'ineq', 'fun': lambda par: par[1]},
    #         {'type': 'ineq', 'fun': lambda par: par[0]})

    bnds = ((0, 1), (0, 1), (300, 3000))

    result = minimize(target_function, [0.1, 0.5, 1000],  bounds=bnds)

    return result.x

def t_estimate_simplified(wavelength, intensity_meas):
    wavelength = np.array(wavelength, dtype='float')
    intensity_meas = np.array(intensity_meas)
    temperature = 1.438775e-2 * (1/(wavelength[1]*1e-9) - 1/(wavelength[0]*1e-9)) / (np.log(intensity_meas[0]/intensity_meas[1]) - 5 * np.log(wavelength[1]/wavelength[0]))
    return temperature