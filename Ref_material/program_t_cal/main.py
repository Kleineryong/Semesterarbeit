import os
import multiprocessing as mp
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import normalize as nr
from scipy.constants import c, h, k
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
currentdir = os.getcwd()
homefolder = os.path.dirname(os.getcwd())

wl = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.82])*10**(-6)
camera_folder = homefolder+"/hyper/camera_parameter/"
exp_case = ["_e005","_e01","_e02","_e03","_e05","_e1"]
DF_QE = pd.read_excel(camera_folder+"CMS22010236.xlsx",'QE')
DF_T = pd.read_excel(camera_folder+"FIFO-Lens_tr.xlsx")


tr_array = np.array(DF_T).transpose()
qe_array = []
for i in range(8):
    qe_array.append(DF_QE.iloc[:, [0,1+i]])
qe_array = np.array(qe_array).transpose(0,2,1)

wl0 = 0.5 * 10 ** (-6)
wl1 = 1 * 10 ** (-6)

def nl_data(data):
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(x - min_value) / (max_value - min_value) for x in data]
    return normalized_data

def lin_int(x,x0,x1,y0,y1):
    y = y0+(y1-y0)*(x-x0)/(x1-x0)
    return y

def GT(temp,wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    result_value = param1/(wavelength**5)/(np.exp(param2/(wavelength*temp))-1)
    return result_value

def integr(wl,f_array,qe_array,a,b,t):
    f_i = len(f_array[0,f_array[0,:]*10**(-9)<=wl])
    qe_i = len(qe_array[0,qe_array[0,:]*10**(-9)<=wl])
    f = lin_int(wl*10**9,f_array[0,f_i-1],f_array[0,f_i],f_array[1,f_i-1],f_array[1,f_i])
    qe = lin_int(wl*10**9,qe_array[0,qe_i-1],qe_array[0,qe_i],qe_array[1,qe_i-1],qe_array[1,qe_i])
    result = f*(a-b*(wl-wl0)/(wl1-wl0))*qe*GT(t,wl)*200
    return result

def process_itg(inten_array,qe_array,tr_array):
    def inte_solve(qe,a,b,t):
        result_f = []
        for i in range(8):
            funct = quad(integr,wl0,wl1,args=(tr_array,qe[i],a,b,t),epsabs = 1e-2, limit=5)[0]
            result_f.append(funct)
        return np.array(result_f)
    popt,cov = curve_fit(inte_solve,qe_array,inten_array,bounds=((0,0,500),(1,1,1958.2)),maxfev=100000)
    return popt[2],popt[0],popt[1]

for idx in range(len(exp_case)):
    experiment_folder = homefolder+"/hyper/data/T1350_0_digital"+exp_case[idx]+"/"
    data = []
    for i in range(8):
        data.append(pd.read_excel(experiment_folder+"digital_value_1350.xlsx","channel_"+str(i),header=None))
    T_ref = np.array(pd.read_excel(experiment_folder+"t_field_1350.xlsx",header=None))
    data = np.array(data)
    target = data[:, 15:25, 15:25]
    t_map = np.zeros((len(target[0]), len(target[0, 0])))
    Ea_map = np.zeros((len(target[0]), len(target[0, 0])))
    Eb_map = np.zeros((len(target[0]), len(target[0, 0])))
    for i in range(len(target[0])):
        for j in range(len(target[0, 0])):
            popt = process_itg(target[:, i, j], qe_array, tr_array)
            t_map[i, j] = popt[0]
            Ea_map[i, j] = popt[1]
            Eb_map[i, j] = popt[2]

    save_dir = homefolder+"/result/BB_change_e/T1350_"+exp_case[idx]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df1 = pd.DataFrame(t_map)
    df1.to_excel(save_dir+"/"+'T.xlsx', index=False)
    df2 = pd.DataFrame(Ea_map)
    df2.to_excel(save_dir + "/" + 'Ea.xlsx', index=False)
    df3 = pd.DataFrame(Eb_map)
    df3.to_excel(save_dir + "/" + 'Eb.xlsx', index=False)
    df4 = pd.DataFrame(t_map-T_ref[15:25, 15:25])
    df4.to_excel(save_dir + "/" + 'T_ab.xlsx', index=False)