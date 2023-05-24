import os
import multiprocessing as mp
import pandas as pd
path = 'G:\\Uni_Files\\Semesterarbeit\\program'
os.chdir(path)
import numpy as np

from camera_parameter import read_camera
from hypothetical import hypothetical
from temperature import t_estimate
from observation import observation
from PIL import Image
import time
from camera_parameter import read_camera
from joblib import Parallel, delayed



#####################################
# parameter definition
# output:
# T1600_data3: absolute value of radiation
# T1600_data3_add_info: emissivity field, temperature_field[k]
# T1600_data3_norm: TBD(maybe digital value)
#####################################
print("Number of processors: ", mp.cpu_count())
image_resolution = [20, 20]                                                 # [pixel] [100 * 100]
diameter_ratio = 0.9                                                        # adjust the visualisation of data_field

temperature_center = 1400                                                   # temperature of the center_area
temperature_background = 50  # set background temperature to 50K as black body
melt_temperature = 1900                                                     # only useful in emissivity_model
temperature_distribution = 'linear'                                       # gaussian / linear / sigmoid

emissivity_type = 'data'                                                    # data (model not available at the moment)
emissivity_set = 3                                                          # which data set is used

explosure_time = 800                                                          # explosure time of camera


QE_address = 'camera_parameter\\CMS22010236.xlsx'
T_address = 'camera_parameter\\FIFO-Lens_tr.xls'

# read camera parameter
cam_param = read_camera.read_cam(QE_address, T_address)
wavelength_set = cam_param['wavelength_set']
cam_efficiency = cam_param['camera_total_efficiency']


# temperature field
t_field = observation.temperature_field(image_resolution, diameter_ratio, temperature_center, temperature_background,
                                        temperature_distribution)

start_time = time.perf_counter()
# real radiation
real_radiation = Parallel(n_jobs=-1)(delayed(observation.radiation_model_conti)(t_field, wavelength_set,
                                                                                emissivity_type, emissivity_set,
                                                                                melt_temperature,
                                                                                temperature_center) for _ in range(1))
# parallel calculation

# real_radiation = observation.radiation_model_conti(t_field, wavelength_set, emissivity_type, emissivity_set,
#                                                    melt_temperature, temperature_center)
# single process


# using the camera model to generate digital value
digital_radiation = observation.camera_model(real_radiation[0]['radiation'], cam_efficiency, explosure_time)

# print(digital_radiation['channel_0'])
end_time = time.perf_counter()
print("calculation time: ", end_time - start_time, " second")

a = read_camera.save_file(t_field, real_radiation[0]['radiation'], digital_radiation, temperature_center, emissivity_set)

