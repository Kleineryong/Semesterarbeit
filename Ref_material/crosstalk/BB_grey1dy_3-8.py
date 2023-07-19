import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, h, k
from scipy.optimize import curve_fit
import tifffile as tif
import time
currentdir = os.getcwd()
homefolder = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
time0 = time.perf_counter()
from joblib import Parallel, delayed
print(currentdir)

wl0 = 0.5 * 10 ** (-6)
wl1 = 0.9 * 10 ** (-6)
param1 = h * 2 * c ** 2
param2 = h * c / k
wl = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.82])*10**(-6)
WL = ["0.548","0.585","0.625","0.663","0.706","0.744","0.785","0.824"]
T_array = np.array([700, 750, 800, 850, 900, 900, 950, 950, 950, 1000, 1000, 1050, 1075, 1085]) + 273.15
camera_folder = homefolder+"/13_Databank/"
Anti_CT = np.array(pd.read_excel(camera_folder+"CT_cor.xlsx"))
sens_fac_a = np.array([11634.066905901529, 31126.869258121664, 55084.34656585057, 112398.30988257349, 108401.90035477096, 122609.03225281756, 160570.92010758465, 146749.15839857998])
sens_fac_b = np.array([46.4382275305586, 137.37680410700113, 175.192679482073, 433.11033989841815, 111.379692118546, -448555.4502610667, -71.35715514970161, -44.58634294203909])

def getfile(homefolder, experiment_name):
    RAWdata_folder = homefolder + "/" + experiment_name
    os.chdir(RAWdata_folder)
    file_list = list(filter(lambda x: x[-5:] == ".tiff", os.listdir(RAWdata_folder)))
    file_list.sort()
    return file_list
def get_data(file_list):
    data = []
    for img_series in range(round(len(file_list) / 9)):
        data_betw = []
        for idx_chl in range(9):
            cI = tif.imread(file_list[img_series * 9 + idx_chl])
            data_betw.append(cI)
        data.append(data_betw)
    return np.asarray(data)

def gaussian(x, sigma):
    return np.exp(-(x**2) / (2 * (sigma**2)))

def bilateral_filter(image, diameter, sigma_space, sigma_intensity):
    filtered_image = np.zeros(image.shape)
    pad = diameter // 2
    padded_image = np.pad(image, pad, mode='reflect')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y+diameter, x:x+diameter]
            intensity_diff = region - padded_image[y+pad, x+pad]
            space_weights = gaussian(np.arange(-pad, pad+1), sigma_space)[:, None] * gaussian(np.arange(-pad, pad+1), sigma_space)[None, :]
            intensity_weights = gaussian(intensity_diff, sigma_intensity)
            weights = space_weights * intensity_weights
            weights /= weights.sum()
            filtered_image[y, x] = (region * weights).sum()
    return filtered_image

def CT_cor(image):    # image in [ch,y,x]
    new_image = []
    for i in range(8):
        image_new = np.zeros((len(image[0])-2,len(image[0,0])-2))
        for j_idx in range(len(image_new)):
            for k_idx in range(len(image_new[0])):
                j = j_idx+1
                k = k_idx+1
                if i == 0:
                    array = [image[i,j,k],image[i+1,j,k],image[i+2,j,k],image[i+3,j,k+1],image[i+4,j,k+1],image[i+5,j-1,k+1],image[i+6,j-1,k],image[i+7,j-1,k],image[i+8,j,k]]
                if i == 1:
                    array = [image[i-1,j,k],image[i,j,k],image[i+1,j,k],image[i+2,j,k+1],image[i+3,j,k+1],image[i+4,j,k+1],image[i+5,j,k],image[i+6,j,k],image[i+7,j,k]]
                if i == 2:
                    array = [image[i-2,j,k],image[i-1,j,k],image[i,j,k],image[i+1,j,k],image[i+2,j,k],image[i+3,j,k],image[i+4,j,k],image[i+5,j,k],image[i+6,j,k]]
                if i == 3:
                    array = [image[i-3,j,k-1],image[i-2,j,k-1],image[i-1,j,k],image[i,j,k],image[i+1,j,k],image[i+2,j,k],image[i+3,j,k],image[i+4,j,k-1],image[i+5,j,k]]
                if i == 4:
                    array = [image[i-4,j,k-1],image[i-3,j,k-1],image[i-2,j,k],image[i-1,j,k],image[i,j,k],image[i+1,j-1,k],image[i+2,j-1,k],image[i+3,j-1,k-1],image[i+4,j,k]]
                if i == 5:
                    array = [image[i-5,j+1,k-1],image[i-4,j,k],image[i-3,j,k],image[i-2,j,k],image[i-1,j+1,k],image[i,j,k],image[i+1,j,k],image[i+2,j,k-1],image[i+3,j+1,k]]
                if i == 6:
                    array = [image[i-6,j+1,k],image[i-5,j,k],image[i-4,j,k],image[i-3,j,k],image[i-2,j+1,k],image[i-1,j,k],image[i,j,k],image[i+1,j,k],image[i+2,j+1,k]]
                if i == 7:
                    array = [image[i-7,j+1,k],image[i-6,j,k],image[i-5,j,k],image[i-4,j,k+1],image[i-3,j+1,k+1],image[i-2,j,k+1],image[i-1,j,k],image[i,j,k],image[i+1,j+1,k]]
                image_new[j_idx,k_idx]= Anti_CT[:,i] @ array
        new_image.append(image_new)
    return np.array(new_image)
def GT(temp,wavelength):
    result_value = param1/(wavelength**5)/(np.exp(param2/(wavelength*temp))-1)
    return result_value

def targetF(wl,a,t):
    result = []
    for i in range(8):
        result.append((a)*GT(t,wl[i]))
    return np.array(result[3:])

def process(i):
    threshold = 80
    wl = np.array([0.548, 0.586, 0.628, 0.667, 0.704, 0.743, 0.786, 0.82])*10**(-6)
    exp_cases = ["700","750","800","850","900.1","900","950","950.3","1000","1050","1075","1085"]
    T_cases = np.array([700,750,800,850,900.1,900,950,950.3,1000,1050,1075,1085])+273
    expor_cases = [3.51,3.51,0.99,1.01,0.1,0.39,0.39,0.31,0.21,0.1,0.1,0.1]
    experiment_folder= "/10_MusTAM_EXP\Try_with_offaxisFactor/V0_bb_"
    ex_factor = expor_cases[i]*emi
    t_soll = T_cases[i]
    data = get_data(getfile(homefolder,experiment_folder+exp_cases[i]))
    fig = data[0]
    print(np.shape(fig))
    denoise = []
    for j in range(9):
        bilateral = bilateral_filter(fig[j,:,:],11,30,30)
        denoise.append(bilateral)
    denoise = np.array(denoise)
    noise = np.mean(denoise[:,0:5,0:5],axis=(1,2)).reshape((9,1,1))
    print(np.shape(denoise))
    ACT_fig = CT_cor((denoise - noise))/ex_factor
    leny = len(ACT_fig[0])
    lenx = len(ACT_fig[0,0])
    T_map = np.zeros((leny,lenx))+273
    Ea_map = np.zeros((leny,lenx))
    for j in range(leny):
        for k in range(lenx):
            if fig[7,j+1,k+1] > threshold:
                PV =ACT_fig[:,j,k]*sens_fac_a+sens_fac_b
                popt,cov = curve_fit(targetF,wl,PV[3:],bounds=((0,500),(1,1958.2)),maxfev=100000)
                T_map[j,k]=popt[1]
                Ea_map[j,k]=popt[0]
    im = plt.imshow(T_map)
    plt.colorbar(im, orientation='vertical')
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    plt.rcParams.update({'font.size': 16})
    plt.title("Soll_"+str(t_soll)+" K")
    plt.savefig(now_dir + "/"+str(t_soll)+".jpg")
    plt.clf()
    df1 = pd.DataFrame(T_map)
    df2 = pd.DataFrame(Ea_map)
    with pd.ExcelWriter(now_dir+"/"+str(t_soll)+".xlsx") as writer:
        df1.to_excel(writer,sheet_name = "T",index = False)
        df2.to_excel(writer, sheet_name="Ea", index=False)
    return

exp_cases = ["700","750","800","850","900.1","900","950","950.3","1000","1050","1075","1085"]
T_cases = np.array([700,750,800,850,900.1,900,950,950.3,1000,1050,1075,1085])+273
expor_cases = [3.51,3.51,0.99,1.01,0.1,0.39,0.39,0.31,0.21,0.1,0.1,0.1]
experiment_folder= "/10_MusTAM_EXP\Try_with_offaxisFactor/V0_bb_"
for idx_emi in range(3):
    emi = 1.0+(idx_emi)**2
    now_dir = currentdir + "/3-8_1d_emi" + str(1/emi)
    if not os.path.exists(now_dir):
        os.makedirs(now_dir)
    result = Parallel(n_jobs=-1)(delayed(process)(x) for x in range(len(exp_cases)))

time1 = time.perf_counter()
print(time1-time0)