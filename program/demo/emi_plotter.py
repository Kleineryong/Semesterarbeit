import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def emi_plotter():
    emi_field = pd.read_excel("t_bias.xlsx", header=None)
    emi_field[emi_field<-0.05] = -0.05
    plt.imshow(emi_field, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Emissivity_map')
    plt.savefig('t_bias' + '.jpg')
    return 0


if 1:
    emi_plotter()
