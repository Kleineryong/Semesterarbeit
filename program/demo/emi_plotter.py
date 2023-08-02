import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd



def emi_iron():
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)

    emi_liquid = np.loadtxt(os.path.join(homefolder, 'hypothetical', 'emissivity_iron_liquid.txt'))
    emi_solid = np.loadtxt(os.path.join(homefolder, 'hypothetical', 'emissivity_iron_solid.txt'))

    emi_temp = np.loadtxt(os.path.join(homefolder, 'hypothetical', 'emissivity_iron_temp.txt'))

    # plot emi_wl
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = fig.add_subplot()
    ax.plot(emi_liquid[:, 0], emi_liquid[:, 1], 'r-', label='Liquid at melt point')
    ax.plot(emi_solid[:, 0], emi_solid[:, 1], 'b-', label='Solid at melt point')

    ax.set_xlabel('Wavelength[nm]', fontsize=20)
    ax.set_ylabel('Emissivity[nm]', fontsize=20)

    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig('emissivity_wl_iron.jpg')
    plt.clf()
    plt.close(fig)

    # plot emi-temp
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = fig.add_subplot()
    ax.plot(emi_temp[:, 0], emi_temp[:, 1], 'ro')
    plt.ylim(0.3, 0.5)

    ax.set_xlabel('Temperature[K]', fontsize=20)
    ax.set_ylabel('Emissivity[nm]', fontsize=20)
    plt.tight_layout()

    # ax.legend(fontsize=20)
    plt.savefig('emissivity_temp_iron.jpg')
    plt.clf()
    plt.close(fig)
    return 0


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
    # emi_plotter()
    emi_iron()