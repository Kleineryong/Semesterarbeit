import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def emi_plotter():
    emi_field = pd.read_excel("emi.xlsx", header=None)
    emi_field[emi_field>1] = 1
    plt.imshow(emi_field, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Emissivity_map')
    plt.savefig('emi_cal' + '.jpg')
    return 0


if 1:
    emi_plotter()
