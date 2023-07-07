from camera_parameter import read_camera
import os
import time
import matplotlib.pyplot as plt
from scipy.constants import c, h, k
import numpy as np
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

def camera_frequency_response():
    currentdir = os.getcwd()
    homefolder = os.path.dirname(currentdir)

    qe_address = os.path.join(homefolder, 'camera_parameter', 'CMS22010236.xlsx')
    tr_address = os.path.join(homefolder, 'camera_parameter', 'FIFO-Lens_tr.xls')

    cam_data = read_camera.read_cam(qe_address, tr_address)
    plt.rcParams.update({'font.size': 15})
    fig_1, ax1 = plt.subplots()
    lin1 = ax1.plot(cam_data['wavelength_set'], cam_data['transparency'])
    ax1.set_xlabel('wavelength[nm]')
    ax1.set_ylabel('Gain')
    plt.tight_layout()
    plt.savefig("tr_frequency_response.jpg")
    plt.clf()

    tr_fun = interp1d(cam_data['wavelength_set'], cam_data['transparency'])
    tr = tr_fun(cam_data['wavelength_set'])

    fig_2, ax2 = plt.subplots(figsize=(9,6))

    ch_0 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 0], label='channel_1')
    ch_1 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 1], label='channel_2')
    ch_2 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 2], label='channel_3')
    ch_3 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 3], label='channel_4')
    ch_4 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 4], label='channel_5')
    ch_5 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 5], label='channel_6')
    ch_6 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 6], label='channel_7')
    ch_7 = ax2.plot(cam_data['wavelength_set'], cam_data['quantum_efficiency'][:, 7], label='channel_8')

    ax2.set_xlabel("Wavelength[nm]")
    ax2.set_ylabel("Quantum efficiency")
    plt.tight_layout()
    ax2.legend(loc='upper right')
    plt.savefig('quantum_efficiency.jpg')
    plt.clf()

    intensity = black_body_radiation(1000, (cam_data['wavelength_set'] * 1e-9))

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 生成示例数据
    x = cam_data['wavelength_set']
    y1 = cam_data['quantum_efficiency'][:, 0] * tr
    y2 = intensity
    y3 = intensity * cam_data['quantum_efficiency'][:, 0] * tr

    # 绘制第一个数据集和对应的坐标轴
    color1 = 'tab:red'
    ax1.set_xlabel('Wavelength[nm]')
    ax1.set_ylabel('Camera efficiency', color=color1)
    ax1.plot(x, y1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # 创建第二个坐标轴并绘制第二个数据集
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Incoming radiation', color=color2)
    ax2.plot(x, y2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 创建第三个坐标轴并绘制第三个数据集
    ax3 = ax1.twinx()
    color3 = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))  # 将第三个坐标轴移动到右侧
    ax3.set_ylabel('Received radiation', color=color3)
    ax3.plot(x, y3, color=color3)
    ax3.tick_params(axis='y', labelcolor=color3)

    # 调整图的布局
    fig.tight_layout()

    # 保存图像
    plt.savefig('received_radiation.jpg')
    return 0


def black_body_radiation(temperature, wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    return param1/(wavelength**5)/(np.exp(param2/(wavelength*temperature))-1)


if 1:
    start_time = time.perf_counter()
    camera_frequency_response()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")
