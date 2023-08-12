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

    emi_model = ['result_v060_lin_square', 'result_v060_lin', 'result_v060_exp']
    emi_model_save = ['lin_square', 'linear', 'exp']

    for i in range(len(emi_model)):
        t_raw, emi_raw, t_ori= read_result(os.path.join(result_dir, emi_model[i]))
        save_result(t_raw, emi_raw, t_ori, emi_model_save[i], save_dir)

    return 0


def save_result(t_raw, emi_raw, t_ori, emi_model_name, save_dir):
    dir_list = ['T3500_0_digital',
                'T3500_5_digital', 'T3500_21_digital', 'T3500_22_digital',
                'T3500_32_digital', 'T3500_33_digital']
    dir_save_list = ['0', '5', '21', '22', '32', '33']
    for i in range(len(dir_save_list)):
        if not os.path.exists(os.path.join(save_dir, dir_save_list[i], "T3500")):
            os.mkdir(os.path.join(save_dir, dir_save_list[i], "T3500"))
        if not os.path.exists(os.path.join(save_dir, dir_save_list[i], "T3500", emi_model_name)):
            os.mkdir(os.path.join(save_dir, dir_save_list[i], "T3500", emi_model_name))

        plt.figure(figsize=(7, 6), dpi=400)

        plt.imshow(np.abs(t_raw[dir_list[i]]), cmap='coolwarm', vmin=0, vmax=0.1)
        cbar = plt.colorbar()
        plt.xlabel('X_position', fontsize=20)
        plt.ylabel('Y_position', fontsize=20)
        plt.title('Rel. temperature difference', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, dir_save_list[i], "T3500", emi_model_name, 'T_bias.jpg'))
        plt.clf()
        plt.close()

        plt.figure(figsize=(7, 6), dpi=400)

        plt.imshow(emi_raw[dir_list[i]], cmap='viridis', vmin=0, vmax=1)
        cbar = plt.colorbar()
        plt.xlabel('X_position', fontsize=20)
        plt.ylabel('Y_position', fontsize=20)
        plt.title('Emissivity', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, dir_save_list[i], "T3500", emi_model_name, 'emi_cal.jpg'))
        plt.clf()
        plt.close()

        plt.figure(figsize=(7, 6), dpi=400)
        plt.imshow(np.abs(t_ori[dir_list[i]]), cmap='inferno', vmin=2500, vmax=3500)
        cbar = plt.colorbar()
        plt.xlabel('X_position', fontsize=20)
        plt.ylabel('Y_position', fontsize=20)
        plt.title('Cal. temperature', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, dir_save_list[i], "T3500", emi_model_name, 'T_cal.jpg'))
        plt.clf()
        plt.close()


def read_result(result_dir):
    dir_list = ['T3500_0_digital',
                'T3500_5_digital', 'T3500_21_digital', 'T3500_22_digital',
                'T3500_32_digital', 'T3500_33_digital']
    t_raw = {}
    emi_raw = {}
    t_ori = {}
    for result in dir_list:
        t_raw[result] = np.array(pd.read_excel(os.path.join(result_dir, result, 't_bias.xlsx'), header=None))
        emi_raw[result] = np.array(pd.read_excel(os.path.join(result_dir, result, 'emi_cal_3500.xlsx'), header=None))
        t_ori[result] = np.array(pd.read_excel(os.path.join(result_dir, result, 't_cal_3500.xlsx'), header=None))
        t_raw[result] = -1 * t_raw[result]
    return t_raw, emi_raw, t_ori


if 1:
    start_time = time.perf_counter()

    result_demo()

    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")