import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


def result_visual_analyze():
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)

    data = read_excel(os.path.join(homefolder, 'results'))

    label = ['linear', 'linear square', 'quadratic', 'exponential',
                  'mixed']

    fig = plt.figure(figsize=(8, 4), dpi=400)
    i = 0
    for key, data_temp in data.items():
        plt.plot(data_temp['diff_rel'][0:11], label=label[i])
        i=i+1

    plt.xticks(range(11), ['Black body', 'Iron', 'model 1', 'model 2', 'model 3', 'model 4', 'model 5', 'model 6', 'model 7', 'model 8', 'model 9'])
    plt.xlabel('Material')
    plt.ylabel('Rel. difference')
    plt.legend()
    plt.tight_layout()
    plt.savefig('diff_rel.jpg')
    fig = plt.close()
    plt.clf()

    fig = plt.figure(figsize=(8, 4), dpi=400)
    i = 0
    for key, data_temp in data.items():
        plt.plot(data_temp['diff_std'][0:11], label=label[i])
        i = i + 1

    plt.xticks(range(11),
               ['Black body', 'Iron', 'model 1', 'model 2', 'model 3', 'model 4', 'model 5', 'model 6', 'model 7',
                'model 8', 'model 9'])
    plt.xlabel('Material')
    plt.ylabel('SD')
    plt.legend()
    plt.tight_layout()
    plt.savefig('diff_std.jpg')
    fig = plt.close()
    plt.clf()

    return 0


def read_excel(home_dir):
    result_dir = ['result_v024_lin', 'result_v030_lin_square', 'result_v024_lin_square', 'result_v024_exp',
                  'result_v040_lin_square_exp']
    data = {}
    for model in result_dir:
        data[model] = {}
        data[model]['diff_rel'] = []
        data[model]['diff_std'] = []
        sheets = pd.read_excel(os.path.join(home_dir, model, 'analyze.xlsx'), sheet_name=None)
        for sheet_name, sheet_data in sheets.items():
            data[model]['diff_rel'].append(sheet_data.iloc[0, 1])
            data[model]['diff_std'].append(sheet_data.iloc[1, 1])

    return data


if 1:
    start_time = time.perf_counter()

    result_visual_analyze()

    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")