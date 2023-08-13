import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt


def temp_distribution(distribution):
    ratio = 0.9
    temperature_center = 3500
    temperature_background = 2500
    resolution = [50, 50]
    field_1 = temperature_field(resolution, ratio, temperature_center, temperature_background, distribution)

    fig = plt.figure(figsize=(8, 6), dpi=400)
    plt.imshow(field_1, cmap='inferno')
    plt.colorbar()
    plt.xlabel('X_position', fontsize=20)
    plt.ylabel('Y_position', fontsize=20)
    plt.title('Temperature_map', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join('t_field_3500' + distribution + '.jpg'))
    plt.clf()

    # fig, axs = plt.subplots()
    #
    # y_t = field_1[len(field_1) // 2]
    #
    # x_radiation = np.arange(resolution[0]) + 1
    # axs.plot(x_radiation, y_t)
    # axs.set(title='temperature distribution', ylabel='temperature')
    # plt.tight_layout()
    # plt.savefig(os.path.join('t_field_' + distribution + '_plot.jpg'))
    return 0


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
                                    sigmoid((-((i - row_center)**2 + (j - colum_center)**2) + radius**2) /
                                            radius**2 * 1000)))
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


def sigmoid(x):
    return 1. / (1. + math.exp(-x/100))


def distribution(x, sigma):
    result = 1 / math.exp(x / (2 * sigma**2))
    return result


if 1:
    start_time = time.perf_counter()

    # temp_distribution('sigmoid')
    temp_distribution('linear')
    # temp_distribution('gaussian')

    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")