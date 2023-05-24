import multiprocessing as mp
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time

from read_data import read_crop_data
from algorithm import process_itg, image_to_array
from MSI_plot import MSI_plot


def main(DF_QE,tr_array, homefolder, experiment_folder, save_dir):
    time0 = time.perf_counter()
    correct_data = read_crop_data(homefolder, experiment_folder,inte_sens)
    mat_array, qe_array = image_to_array(DF_QE, correct_data, threshold)
    
    result = Parallel(n_jobs=-1)(delayed(process_itg)(x, correct_data, mat_array,qe_array,tr_array) for x in range(8))

    time2 = time.perf_counter()
    result = np.array(result)
    print(np.shape(result))

   
    print("run_time",time2-time0)

    MSI_plot(correct_data, mat_array, result, save_dir)

    return


if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())
    wl = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.826])*10**(-6)
    inte_sens = np.array([0.45758346, 0.47678455, 0.50376139, 0.51965259, 0.50709045, 0.50204587, 0.49392519, 0.48651687])
    homefolder = os.getcwd()

    camera_folder = homefolder+"/input/00_Databank/"

    DF_F = pd.read_excel(camera_folder+"CMS22010236.xlsx",'Filter')
    DF_QE = pd.read_excel(camera_folder+"CMS22010236.xlsx",'QE')
    DF_T = pd.read_excel(camera_folder+"FIFO-Lens_tr.xlsx")
    tr_array = np.array(DF_T).transpose()
    threshold = 90
    experiment_series = [800]#[600,625,650,675,700,725,750,775,800]
    for i in experiment_series:
        experiment_folder = "input\\V1_" + str(i) + "_oven"
        save_dir = homefolder + "output\\V1_" + str(i) + "_oven"
        main(DF_QE, tr_array, homefolder, experiment_folder, save_dir)





