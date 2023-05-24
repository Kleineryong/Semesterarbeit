import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def MSI_plot(cor_data, mat_array, result, save_dir):
    plot_data = np.zeros(np.shape(cor_data[:,:,0]))+1
    k_map = np.zeros(np.shape(cor_data[:,:,0]))+0.0001
    b_map = np.zeros(np.shape(cor_data[:,:,0]))+0.0001
    n = 0
    for i in range(len(plot_data)):
        for j in range(len(plot_data[1])):
            for k in range(len(mat_array)):
                if(mat_array[k,0]==i and mat_array[k,1]==j):
                    if(result[k,0]>1400):
                        n += 1
                    plot_data[i,j]=result[k,0]
                    k_map[i,j]=result[k,1]
                    b_map[i,j]=result[k,2]

    plt.clf()
    print(plot_data.max())
    im = plt.imshow(plot_data)
    plt.colorbar(im, orientation='vertical')
    plt.clim(1200, 1380)
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    plt.rcParams.update({'font.size': 16})
    plt.title("temperature/K")

    plt.savefig(save_dir+"/T.png")
    df1 = pd.DataFrame(plot_data)
    df2 = pd.DataFrame(k_map)
    df3 = pd.DataFrame(b_map)
    with pd.ExcelWriter(save_dir + "/800.xlsx") as writer:
        df1.to_excel(writer, sheet_name="T", index=False)
        df2.to_excel(writer, sheet_name="a", index=False)
        df3.to_excel(writer, sheet_name="b", index=False)

    return