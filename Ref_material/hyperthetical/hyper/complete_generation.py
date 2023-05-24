import os
from observation import observation


#####################################
# set the path to the program
#####################################
path = 'G:\\Uni_Files\\Semesterarbeit\\program'
os.chdir(path)

#####################################
# parameter definition
# output:
# T1600_data3: absolute value of radiation
# T1600_data3_add_info: emissivity field, temperature_field[k]
# T1600_data3_norm: TBD(maybe digital value)
#####################################
channel_set = ['548', '586', '628', '667', '704', '743', '786', '826']      # wavelength of observation [nm]
image_resolution = [100, 100]                                               # [pixel]
diameter_ratio = 0.9                                                        # adjust the visualisation of data_field

temperature_center = 1700                                                   # temperature of the center_area
temperature_background = 50  # set background temperature to 50K as black body
melt_temperature = 1900                                                     # only useful in emissivity_model
temperature_distribution = 'gaussian'                                       # gaussian / linear / sigmoid

emissivity_type = 'data'                                                    # data/model
emissivity_set = 3                                                          # which data set is used

#####################################
# generate temperature field and radiation field
#####################################
result = observation.radiation_model(channel_set, emissivity_type, emissivity_set, melt_temperature, image_resolution,
                                       diameter_ratio, temperature_center, temperature_background,
                                       temperature_distribution)

t_field_real = result['temperature']
e_field_real = result['emissivity']

del result