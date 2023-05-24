import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from temperature import read_file_raw_radiation
from temperature import t_estimate

path = 'G:\\Uni_Files\\Semesterarbeit\\program'
os.chdir(path)

address = 'data\T1500_data4\T1500_channel_826.tiff'

file_address = 'data\T1500_data4'
data = read_file.read(file_address)

test = read_file.t_field_cal(data)
# print(test)

nrows, ncols = test.shape
x, y = np.meshgrid(range(ncols), range(nrows))

# print(x)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x.flatten(), y.flatten(), data.flatten())
# plt.show()
fig = plt.figure(figsize=(15, 11))
ax = plt.axes(projection="3d")
ax.scatter3D(x, y, test, color="green", s=5)
plt.title("estimated temperature")
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('estimated_temperature')

# show plot
plt.show()