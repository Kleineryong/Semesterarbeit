from camera_parameter import read_camera
import os
import time
import matplotlib.pyplot as plt


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

    return 0


if 1:
    start_time = time.perf_counter()
    camera_frequency_response()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")
