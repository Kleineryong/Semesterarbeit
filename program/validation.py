import numpy as np
import os
from observation import observation
from temperature import read_file_raw_radiation
import matplotlib.pyplot as plt

# set parameter
channel_set = ['548', '586', '628', '667', '704', '743', '786', '826']
image_resolution = [100, 100]
temperature_center = 1600
temperature_background = 50  # set background temperature to 50K as black body
diameter_ratio = 0.9
emissivity_type = 'data'        #data/model
melt_temperature = 1900
emissivity_set = 3

# generate temperature field and radiation field
result = observation.radiation_model(channel_set, emissivity_type, emissivity_set, melt_temperature, image_resolution,
                                       diameter_ratio, temperature_center, temperature_background, 'gaussian')

t_field_real = result['temperature']
del result
# calculate temperature

path = os.path.join('data', ('T' + str(temperature_center) + '_' + emissivity_type + str(emissivity_set)))

data = read_file_raw_radiation.read(path)
t_field_cal = read_file_raw_radiation.t_field_cal(data)

nrows, ncols = t_field_cal.shape
x, y = np.meshgrid(range(ncols), range(nrows))

fig = plt.figure(figsize=(15, 11))
ax = plt.axes(projection="3d")
ax.scatter3D(x, y, t_field_real, color="green", s=1, label='real')
ax.scatter3D(x, y, t_field_cal, color='red', s=1, label='cal')
plt.title("temperature")
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('temperature[k]')

# show plot
plt.show()