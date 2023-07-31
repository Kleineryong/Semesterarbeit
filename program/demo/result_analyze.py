import os
import numpy as np
import time
import pandas as pd
import openpyxl


def result_analyze(result_dir):
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)

    # read real temperature field and choose the focused area.
    t_field_raw = np.array(pd.read_excel(os.path.join(homefolder, 'data', 'T1900_5_digital', 't_field_1900.xlsx'),
                                         header=None))
    t_index = t_field_raw > 1500
    t_true = t_field_raw[t_index]

    # read calculated results
    t_result = read_result(os.path.join(homefolder, 'results', result_dir), t_index)

    # calculate difference
    t_diff_abs = {}
    t_diff_rel = {}
    t_diff_std = {}
    t_diff_max = {}
    t_diff_min = {}
    for key in t_result.keys():
        t_diff_abs[key] = np.average(t_result[key] - t_true)
        t_diff_rel[key] = np.average(np.abs(t_result[key] / t_true - 1))
        t_diff_std[key] = np.std(t_result[key] - t_true)
        t_diff_max[key] = np.max(np.abs(t_result[key] / t_true - 1))
        t_diff_min[key] = np.min(np.abs(t_result[key] / t_true - 1))

    save_result(os.path.join(homefolder, 'results', result_dir), t_diff_abs, t_diff_rel, t_diff_std, t_diff_max, t_diff_min)
    return 0


def save_result(result_dir, t_diff_abs, t_diff_rel, t_diff_std, t_diff_max, t_diff_min):
    workbook_digit = openpyxl.Workbook()
    worksheet_digit = workbook_digit.active

    for data_set in t_diff_abs.keys():
        worksheet_digit = workbook_digit.create_sheet(str(data_set))
        temp = t_diff_abs[data_set]
        worksheet_digit.append(['diff_abs', t_diff_abs[data_set]])
        worksheet_digit.append(['diff_rel', t_diff_rel[data_set]])
        worksheet_digit.append(['diff_std', t_diff_std[data_set]])
        worksheet_digit.append(['diff_max', t_diff_max[data_set]])
        worksheet_digit.append(['diff_min', t_diff_min[data_set]])

    workbook_digit.remove(workbook_digit['Sheet'])
    workbook_digit.save(os.path.join(result_dir, 'analyze.xlsx'))
    return 0


def read_result(result_dir, t_index):
    dir_list = ['T1900_0_digital', 'T1900_1_digital', 'T1900_21_digital', 'T1900_22_digital', 'T1900_23_digital',
                'T1900_24_digital', 'T1900_25_digital', 'T1900_26_digital', 'T1900_31_digital',
                'T1900_32_digital', 'T1900_33_digital', 'T1900_34_digital']
    result_raw = {}
    for result in dir_list:
        result_raw[result] = np.array(pd.read_excel(os.path.join(result_dir, result, 't_cal_1900.xlsx'), header=None))[t_index]
    return result_raw


if 1:
    start_time = time.perf_counter()
    result_dir = 'result_v040_lin_square_exp'
    result_analyze(result_dir)

    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")