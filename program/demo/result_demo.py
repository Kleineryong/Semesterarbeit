import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties


def result_demo():
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)

    result_dir = os.path.join(homefolder, 'results')
    save_dir = 'F:\\Uni_Files\\Semesterarbeit\\Thesis\\LBAMStudentThesis\\figures\\raw_data'

    emi_model = ['result_v040_lin_square_exp', 'result_v030_lin_square', 'result_v024_lin', 'result_v024_lin_square', 'result_v024_exp']
    emi_model_save = ['mix', 'lin_square', 'linear', 'quad', 'exp']

    for i in range(len(emi_model)):
        t_raw, emi_raw = read_result(os.path.join(result_dir, emi_model[i]))
        save_result(t_raw, emi_raw, emi_model_save[i], save_dir)

    return 0


def save_result(t_raw, emi_raw, emi_model_name, save_dir):
    dir_list = ['T1900_0_digital',
                'T1900_5_digital', 'T1900_21_digital', 'T1900_22_digital', 'T1900_23_digital',
                'T1900_24_digital', 'T1900_25_digital', 'T1900_26_digital', 'T1900_31_digital',
                'T1900_32_digital', 'T1900_33_digital', 'T1900_34_digital']
    dir_save_list = ['0', '5', '21', '22', '23', '24', '25', '26', '31', '32', '33', '34']
    for i in range(len(dir_save_list)):
        if not os.path.exists(os.path.join(save_dir, dir_save_list[i], emi_model_name)):
            os.mkdir(os.path.join(save_dir, dir_save_list[i], emi_model_name))

        plt.figure(figsize=(8, 6), dpi=400)

        plt.imshow(t_raw[dir_list[i]], cmap='coolwarm')
        cbar = plt.colorbar()
        plt.xlabel('X_position', fontsize=20)
        plt.ylabel('Y_position', fontsize=20)
        plt.title('Rel. temperature difference', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, dir_save_list[i], emi_model_name, 'T_bias.jpg'))
        plt.clf()
        plt.close()

        plt.figure(figsize=(8, 6), dpi=400)

        plt.imshow(emi_raw[dir_list[i]], cmap='viridis')
        cbar = plt.colorbar()
        plt.xlabel('X_position', fontsize=20)
        plt.ylabel('Y_position', fontsize=20)
        plt.title('Emissivity', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, dir_save_list[i], emi_model_name, 'emi_cal.jpg'))
        plt.clf()
        plt.close()

def read_result(result_dir):
    dir_list = ['T1900_0_digital',
                'T1900_5_digital', 'T1900_21_digital', 'T1900_22_digital', 'T1900_23_digital',
                'T1900_24_digital', 'T1900_25_digital', 'T1900_26_digital', 'T1900_31_digital',
                'T1900_32_digital', 'T1900_33_digital', 'T1900_34_digital']
    t_raw = {}
    emi_raw = {}
    for result in dir_list:
        t_raw[result] = np.array(pd.read_excel(os.path.join(result_dir, result, 't_bias.xlsx'), header=None))
        emi_raw[result] = np.array(pd.read_excel(os.path.join(result_dir, result, 'emi_cal_1900.xlsx'), header=None))
        t_raw[result] = -1 * t_raw[result]
    return t_raw, emi_raw


if 1:
    start_time = time.perf_counter()

    result_demo()

    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")