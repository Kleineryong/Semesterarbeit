
import numpy as np
from scipy.constants import c, h, k
from scipy.optimize import curve_fit

from scipy.integrate import quad

wl0 = 0.5 * 10 ** (-6)
wl1 = 1 * 10 ** (-6)

def lin_int(x,x0,x1,y0,y1):
    y = y0+(y1-y0)*(x-x0)/(x1-x0)
    return y

def GT(temp,wavelength):
    param1 = h * 2 * c ** 2
    param2 = h * c / k
    result_value = param1/(wavelength**5)/(np.exp(param2/(wavelength*temp))-1)
    return result_value

def integr(wl,f_array,qe_array,a,b,t):
    f_i = len(f_array[0,f_array[0,:]*10**(-9)<=wl])
    qe_i = len(qe_array[0,qe_array[0,:]*10**(-9)<=wl])
    f = lin_int(wl*10**9,f_array[0,f_i-1],f_array[0,f_i],f_array[1,f_i-1],f_array[1,f_i])
    qe = lin_int(wl*10**9,qe_array[0,qe_i-1],qe_array[0,qe_i],qe_array[1,qe_i-1],qe_array[1,qe_i])
    result = f*(a-b*(wl-wl0)/(wl1-wl0))*qe*GT(t,wl)*100
    return result

def process_itg(idx_arr,cor_data,mat_array,qe_array,tr_array):
    def inte_solve(qe,a,b,t):
        result_f = []
        for i in range(8):
            funct = quad(integr,wl0,wl1,args=(tr_array,qe[i],a,b,t))[0]
            result_f.append(funct)
        return np.array(result_f)
    inten_array = []
    for idx_wl in range(8):
        inten_array.append(cor_data[mat_array[idx_arr,0],mat_array[idx_arr,1],idx_wl])
    popt,cov = curve_fit(inte_solve,qe_array,inten_array,bounds=((0.1,0.01,1000),(1,0.5,1958.2)),maxfev=100000)
    print(idx_arr)
    return popt[2],popt[0],popt[1]

def image_to_array(DF_QE, cor_data, threshold=90):
    qe_array = []
    for i in range(8):
        qe_array.append(DF_QE.iloc[:, [0,1+i]])
    qe_array = np.array(qe_array).transpose(0,2,1)
    print(np.shape(qe_array))

    target_img = cor_data[:,:,7]
    print(np.shape(target_img))

    filter_matr=(target_img>threshold)*1
    filt_mat = target_img*filter_matr
    mat_array = []
    for i in range(len(filt_mat[0])):
        for j in range(len(filt_mat)):
            if filt_mat[j,i]>0:
                mat_array.append([j,i])
    mat_array = np.asarray(mat_array)
    print(len(mat_array)) #### how may pixels will be evaluated

    return mat_array,qe_array




