
"""
Created on Sun Jan 29 16:16:55 2023

@author: Wang Zhaoyong
"""
from hypothetical import hypothetical
from temperature import t_estimate
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    temperature = 1700         # 温度处于1500k - 2000k
    melt_temperature = 1800     # melting-temperature

    lb_wavelength = 500
    ub_wavelength = 900
    interval_wavelength = 5

    emissivity_type = 'data'
    emissivity_data_set = 1

    material = hypothetical.black_body_radiation(temperature, melt_temperature, lb_wavelength, ub_wavelength, interval_wavelength, emissivity_type, emissivity_data_set)

    wavelength = []
    intensity = []

    for i in material:
        wavelength.append(int(i))
        intensity.append(material[i])

    wavelength_sample = ['600','625', '650', '675', '700']
    intensity_sample = []
    for i in range(len(wavelength_sample)):
        intensity_sample.append(material[wavelength_sample[i]])
    result = t_estimate.t_estimate(wavelength_sample, intensity_sample)

    temperature_esti = result[2]
    emissivity_esti = result[1] - result[0] * ((np.array(wavelength_sample, dtype='float')-500)/(900-500))

    temperature_esti_simplified = t_estimate.t_estimate_simplified(wavelength_sample[:2], intensity_sample[:2])

    print(result)
    print("estimated temperature, fitting")
    print(temperature_esti)
    print("estimated temperature, simplified")
    print(temperature_esti_simplified)
    print(emissivity_esti)

    plt.figure(1)
    plt.plot(wavelength, intensity)
    plt.xlabel("wavelength[nm]")
    plt.ylabel("intensity")
    plt.show()