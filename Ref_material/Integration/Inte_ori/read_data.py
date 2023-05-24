import numpy as np
import tifffile as tif
import os

def getfile(homefolder, experiment_name):
    RAWdata_folder = homefolder + "\\" + experiment_name
    os.chdir(RAWdata_folder)
    file_list = list(filter(lambda x: x[-5:] == ".tiff", os.listdir(RAWdata_folder)))
    file_list.sort()
    return file_list
## get data in form (series,x,y,channel)
def get_data(file_list):
    data = []
    for img_series in range(int(len(file_list) / 9)):
        data_betw = []
        for idx_chl in range(8):
            cI = tif.imread(file_list[img_series * 9 + idx_chl])
            data_betw.append(cI)
        data.append(data_betw)
    data = np.transpose(np.asarray(data),(0,2,3,1))
    return np.asarray(data)

def read_crop_data(homefolder, experiment_folder,inte_sens):
    file_list = getfile(homefolder,experiment_folder)
    raw_data = get_data(file_list)
    noise = raw_data[0,0,0]
    fig_data = np.asarray(raw_data[0])
    max_x = 0
    min_x = 999
    max_y = 0
    min_y = 999
    threshold = 70
    for i in range(len(fig_data)):
        for j in range(len(fig_data[0])):
            if(fig_data[i,j,7]>threshold):
                max_x = max(max_x,j)
                min_x = min(min_x,j)
                max_y = max(max_y,i)
                min_y = min(min_y,i)
    cro_data = np.asarray(fig_data[(max(0,min_y-10)):(max_y+10),(max(0,min_x-10)):(max_x+10),:])
    print(np.shape(cro_data))
    cor_data = (cro_data-np.mean(noise))*inte_sens
    return cor_data