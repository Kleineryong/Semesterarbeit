import os
import numpy as np
from hypothetical import hypothetical
import observation

# set file path
path = 'G:\\Uni_Files\\Semesterarbeit\\program'
os.chdir(path)

# set initial parameters
channel_set = ['548', '586', '628', '667', '704', '743', '786', '826']
image_resolution = [100, 100]
temperature_center = 1500
temperature_background = 50  # set background temperature to 50K as black body
diameter_ratio = 0.9
emissivity_type = 'data'        #data/model
melt_temperature = 1800
emissivity_set = 4

# field = np.array(observationmodel.temperature_field(image_resolution, diameter_ratio, temperature_center, temperature_background, 0))
# channel = {}
#
# plank_law_vec = np.vectorize(hypothetical.plank_law)
# emissivity_model_vec = np.vectorize(hypothetical.emissivity_model)
# emissivity_data_vec = np.vectorize(hypothetical.emissivity_data)
# for wavelength in channel_set:
#     if emissivity_type == 'model':
#         emissivity = emissivity_model_vec(field, melt_temperature, float(wavelength))
#     elif emissivity_type == 'data':
#         emissivity = emissivity_data_vec(field, emissivity_set, float(wavelength))
#     channel[str(wavelength)] = plank_law_vec(field, float(wavelength)) * emissivity
# print(channel)
result = observation.radiation_model(channel_set, emissivity_type, emissivity_set, melt_temperature, image_resolution,
                                       diameter_ratio, temperature_center, temperature_background, 0)
# print(result)
# channel = result['radiation']
# # t_field = result['temperature']
# # observation.temperature_plot(channel['548'])
# # observation.temperature_plot(t_field)