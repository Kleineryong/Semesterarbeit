import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

def compare():
    original_data = 'T1896_3_digital'
    original_data_address = os.path.join('data', original_data)

    cal_data = original_data
<<<<<<< HEAD
<<<<<<< HEAD
    cal_data_address = os.path.join('results', 'result_v022_exp', cal_data)
=======
    cal_data_address = os.path.join('result_v022_lin', cal_data)
>>>>>>> parent of 9023769 (V022)
=======
    cal_data_address = os.path.join('result_v022_lin', cal_data)
>>>>>>> parent of 9023769 (V022)

    # read data from excel file
    for file in os.listdir(original_data_address):
        if file.endswith('.xlsx') and file.startswith('t_field'):
            df = pd.read_excel(os.path.join(original_data_address, file), header=None)
            t_target = df.to_numpy()

    for file in os.listdir(original_data_address):
        if file.endswith('.xlsx') and file.startswith('emi_field'):
            df = pd.read_excel(os.path.join(original_data_address, file), header=None)
            emi_target = df.to_numpy()

    for file in os.listdir(cal_data_address):
        if file.endswith('.xlsx') and file.startswith('t_cal'):
            df = pd.read_excel(os.path.join(cal_data_address, file), header=None)
            t_cal = df.to_numpy()

    for file in os.listdir(cal_data_address):
        if file.endswith('.xlsx') and file.startswith('emi_cal'):
            df = pd.read_excel(os.path.join(cal_data_address, file), header=None)
            emi_cal = df.to_numpy()

    t_bias = t_target - t_cal
    emi_bias = emi_target - emi_cal

    ######### save fig
    # t_cal
    plt.imshow(t_cal, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Temperature_cal')
    plt.savefig(os.path.join(cal_data_address, 'T_cal.jpg'))
    plt.clf()

    # emi_cal
    plt.imshow(emi_cal, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('emissivity_calculate')
    plt.savefig(os.path.join(cal_data_address, 'emi_cal.jpg'))
    plt.clf()

    # bias
    file_t = os.path.join(cal_data_address, 't_bias.xlsx')
    workbook_t = openpyxl.Workbook()
    worksheet_t = workbook_t.active

    for row in t_bias:
        worksheet_t.append(list(row))
    workbook_t.save(file_t)

    plt.imshow(t_bias, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('Temperature_bias')
    plt.savefig(os.path.join(cal_data_address, 'T_bias.jpg'))
    plt.clf()

    file_emi = os.path.join(cal_data_address, 'emi_bias.xlsx')
    workbook_emi = openpyxl.Workbook()
    worksheet_emi = workbook_emi.active

    for row in emi_bias:
        worksheet_emi.append(list(row))
    workbook_emi.save(file_emi)

    plt.imshow(emi_bias, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X_position')
    plt.ylabel('Y_position')
    plt.title('emissivity_bias')
    plt.savefig(os.path.join(cal_data_address, 'emi_bias.jpg'))
    plt.clf()



if 1:
    start_time = time.perf_counter()
    compare()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")
