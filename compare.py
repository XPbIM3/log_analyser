import numpy as np
np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)
import glob
import pandas as pd
from lib.TuneParser import *
#USER settings section
import matplotlib
import matplotlib.pyplot as plt
from DataLog import read_raw_data, filter_raw_data, readMslAtLocation, filterValues
matplotlib.rcParams['figure.figsize'] = 15,7




if __name__ == '__main__':
    DAFAULT_INPUT_MSL = "./input/*.msl"
    DEFAULT_COMPARE_MSL = "./compare/*.msl"
    kpa_used_np, rpm_used_np, ve_prediction_np, afr_achieved_np = readMslAtLocation(DAFAULT_INPUT_MSL)

    kpas_np, rpms_np, ves_np, afr_np = filterValues(kpa_used_np, rpm_used_np, ve_prediction_np, afr_achieved_np, (80, 100), (0, 10000))

    plt.scatter(rpms_np, ves_np, s=3, marker='.')

    kpa_used_np, rpm_used_np, ve_prediction_np, afr_achieved_np = readMslAtLocation(DEFAULT_COMPARE_MSL)

    kpas_np, rpms_np, ves_np, afr_np = filterValues(kpa_used_np, rpm_used_np, ve_prediction_np, afr_achieved_np, (80, 100), (0, 10000))

    plt.scatter(rpms_np, ves_np, s=3, marker='.')
    plt.show()

    plt.savefig(f'swipe.png')

