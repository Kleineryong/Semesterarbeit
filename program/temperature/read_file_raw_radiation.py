import os
import tifffile
import numpy as np
from temperature import t_estimate


##############################################
# used to read file generated or collected.
# input: file_address
# output: radiation{'wavelength': field}
##############################################
def read(file_address):
    data = {}
    for file_name in os.listdir(file_address):
        channel_start = file_name.find("_channel_") + 9
        channel_end = file_name.rfind(".tiff")
        channel = file_name[channel_start:channel_end]
        with tifffile.TiffFile(os.path.join(file_address, file_name)) as tif:
            field = np.around(tif.asarray())
        data[channel] = field
    return data


#################################################
# calculate estimated temperature field
# input: data
# output: t_field
#################################################
def t_field_cal(data):
    wavelength = list(data.keys())
    row, colum = data[list(data.keys())[0]].shape
    t_field = np.zeros([row, colum])
    for i in range(row):
        for j in range(colum):
            radiation = []
            for wl in wavelength:
                radiation.append(data[wl][i,j])
            t_field[i, j] = t_estimate.t_estimate(wavelength, radiation)[2]
    return t_field