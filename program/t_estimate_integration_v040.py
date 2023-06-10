import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import warnings
import openpyxl
import multiprocessing as mp
from scipy.constants import c, h, k
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import quad
from scipy.interpolate import interp1d
from joblib import Parallel, delayed


def t_estimate():

    return 0

if 1:
    start_time = time.perf_counter()
    t_estimate()
    end_time = time.perf_counter()
    print("calculation time: ", end_time - start_time, " second")