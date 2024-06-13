import numpy as np
np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)
import glob
import pandas as pd
from lib.TuneParser import *
#USER settings section
import matplotlib.pyplot as plt






SHIFT_AFR = 1
HITS_NEEDED = 10
MARGIN=1.0
GENERATE_AFR_TABLE = True
GENERATE_AXES = False
KPA_BINS_AMOUNT = 16
RPM_BINS_AMOUNT = 16
RPM_QUANT = 100
KPA_QUANT = 2


CONFIG_ATMO_6 = {'KPA_MIN':24, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_ATMO_M104 = {'KPA_MIN':24, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':7000, 'AFR_MIN':15.0, 'AFR_MAX':12.}
CONFIG_ATMO_M20 = {'KPA_MIN':28, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_TURBO_M20 = {'KPA_MIN':28, 'KPA_MAX':160, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.0}
CONFIG_ATMO_VWKR = {'KPA_MIN':20, 'KPA_MAX':100, 'RPM_MIN':600,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_TURBO_SR20 = {'KPA_MIN':24, 'KPA_MAX':200, 'RPM_MIN':600,'RPM_MAX':7000, 'AFR_MIN':14.5, 'AFR_MAX':11.0}

CURRENT_CONFIG = CONFIG_ATMO_M104

############################################


VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
AFR_TABLE_DICT = {'table': 'afrTable', 'xaxis': 'rpmBinsAFR', 'yaxis': 'loadBinsAFR'}
IGN_TABLE_DICT = {'table': 'advTable1', 'xaxis': 'rpmBins2', 'yaxis': 'mapBins1'}
F_NAME = './input/CurrentTune.msq'
RPM_BINS_VE, KPA_BINS_VE, VE_TABLE_PARSED, VE_TABLE_FUNC = getTable(F_NAME,VE_TABLE_DICT)
RPM_BINS_AFR, KPA_BINS_AFR, AFR_TABLE_PARSED, AFR_TABLE_FUNC = getTable(F_NAME,AFR_TABLE_DICT)
RPM_BINS_IGN, KPA_BINS_IGN, IGN_TABLE_PARSED, IGN_TABLE_FUNC = getTable(F_NAME,IGN_TABLE_DICT)


if GENERATE_AXES == False:
    KPA_BINS = KPA_BINS_VE
    RPM_BINS = RPM_BINS_VE
else:
    KPA_BINS = ((np.round(np.linspace(CURRENT_CONFIG['KPA_MIN'],CURRENT_CONFIG['KPA_MAX'], KPA_BINS_AMOUNT)) // KPA_QUANT) * KPA_QUANT).astype(int)
    RPM_BINS = ((np.round(np.linspace(CURRENT_CONFIG['RPM_MIN'],CURRENT_CONFIG['RPM_MAX'], RPM_BINS_AMOUNT)) // RPM_QUANT) * RPM_QUANT).astype(int)


print("bins:")
print(KPA_BINS)
print(RPM_BINS)


VE_TABLE = VE_TABLE_FUNC(RPM_BINS, KPA_BINS)
IGN_TABLE = IGN_TABLE_FUNC(RPM_BINS, KPA_BINS)

if GENERATE_AFR_TABLE:
    AFR_TABLE = np.array([np.linspace(CURRENT_CONFIG['AFR_MIN'], CURRENT_CONFIG['AFR_MAX'], 16)]*16).transpose()
    AFR_TABLE_FUNC = tableInterpolate(RPM_BINS, KPA_BINS, AFR_TABLE)
else:
    AFR_TABLE = AFR_TABLE_FUNC(RPM_BINS, KPA_BINS)


def export(RPMS, KPAS, ZS, fname, dtype=int):
    with open('./templates/VE.table.template', 'r') as t:
        templ = t.read()

    xax = "\n"+("\n".join(list(map(str, RPMS.astype(int)))))+"\n"
    yax = "\n"+("\n".join(list(map(str, KPAS.astype(int)))))+"\n"
    zax = "\n"+str(ZS.astype(dtype)).strip('[ ]').replace('[','').replace(']', '')+"\n"


    templ = templ.replace('_XAXIS_', xax)
    templ = templ.replace('_YAXIS_', yax)
    templ = templ.replace('_ZAXIS_', zax)

    with open(fname, 'w') as t:
        t.write(templ)

'''
def getBinFromValue(value:float, bins:np.ndarray):
    return np.argmin(np.abs(value-bins))
'''




#@profile
def main():



    flist = glob.glob("./input/*.msl")


    pd_frames = []
    for fname in flist:
        fr = pd.read_csv(fname, sep='\t', header=0, skiprows=[0,1,3])
        pd_frames.append(fr)

    data_raw = pd.concat(pd_frames)


    data_raw.AFR = data_raw.AFR.shift(SHIFT_AFR)
    data_raw.Lambda = data_raw.Lambda.shift(SHIFT_AFR)

    data_raw = data_raw.drop(0)

    print(f"Data Total: {len(data_raw)}")
    #vdata = data_raw[(data_raw.Gwarm==100) & (data_raw.RPM>0) & (data_raw.DFCO==0)& (data_raw['rpm/s']>=0) & (data_raw['Accel Enrich']==100) & (data_raw['TPS'] > 0.0)]
    # data = data_raw[(data_raw['rpm/s']>=-1000) & (data_raw.DFCO==0)]
    # data = data_raw[(data_raw['Accel Enrich']==100)]
    data = data_raw

    print(f"Data Used: {len(data)}")


    data['afr_target_func'] = data.apply(lambda row: AFR_TABLE_FUNC(row['RPM'], row['MAP']), axis=1)
    data['corr_coef'] = data.apply(lambda row: row['AFR']/row['afr_target_func'], axis=1)
    data['ve_predicted'] = data.apply(lambda row: row['VE1']*row['corr_coef'], axis=1)


    AFR_achieved = np.full((KPA_BINS.size, RPM_BINS.size), np.nan, dtype=float)
    RPMdot_achieved = np.full((KPA_BINS.size, RPM_BINS.size), np.nan, dtype=float)
    Lambda_achieved = np.full((KPA_BINS.size, RPM_BINS.size), np.nan, dtype=float)
    VE_predicted_weightened = np.zeros((KPA_BINS.size, RPM_BINS.size), dtype=int)
    data_points_amount_map = np.zeros((KPA_BINS.size, RPM_BINS.size), dtype=int)
    ve_predict_std = np.zeros((KPA_BINS.size, RPM_BINS.size), dtype=float)

    kpa_max_step = np.gradient(KPA_BINS).max()
    rpm_max_step = np.gradient(RPM_BINS).max()

    print(kpa_max_step, rpm_max_step)

    for i, kpa_center in enumerate(KPA_BINS):
        for j, rpm_center in enumerate(RPM_BINS):
            kpa_min = kpa_center-kpa_max_step
            kpa_max = kpa_center+kpa_max_step
            rpm_min = rpm_center - rpm_max_step
            rpm_max = rpm_center + rpm_max_step


            pd_local_frame = data[(data.MAP>=kpa_min) & (data.MAP<=kpa_max) & (data.RPM>=rpm_min) & (data.RPM<=rpm_max)]
            data_points_amount = len(pd_local_frame)
            if data_points_amount>0:
                data_points_amount_map[i][j] = data_points_amount
                rpms_np = pd_local_frame['RPM'].to_numpy(dtype=float)
                kpas_np = pd_local_frame['MAP'].to_numpy(dtype=float)
                ves_np = pd_local_frame['ve_predicted'].to_numpy(dtype=float)
                x_dist = np.abs(rpms_np - float(rpm_center)) / float((rpm_max-rpm_min)/2.0)
                y_dist = np.abs(kpas_np - float(kpa_center)) / float((kpa_max-kpa_min)/2.0)
                assert x_dist.max() <= 1.0
                assert y_dist.max() <= 1.0
                weights = 1.0 - (np.sqrt(x_dist**2 + y_dist**2)/np.sqrt(2.0))
                assert np.all(weights >= 0.0)
                weights_norm = weights / np.sum(weights)
                VE_predicted_weightened[i][j] = int(np.round(np.sum(ves_np * weights_norm)))
                ve_predict_std[i][j] = ves_np.std()
                AFR_achieved[i][j] = float(pd_local_frame.AFR.mean())
                Lambda_achieved[i][j] = float(pd_local_frame['Lambda'].mean())
                RPMdot_achieved[i][j] = pd_local_frame['rpm/s'].mean()


    weighted_ve = VE_predicted_weightened.copy()
    # weighted_ve[weighted_ve==0] = VE_TABLE[weighted_ve==0]

    '''
    def saveFig(x_axis: list, y_axis: list, arr: np.ndarray, fname: str):
        arr = np.flipud(arr)
        x_axis_size = arr.shape[0]
        y_axis_size = arr.shape[0]
        x_ticks = np.arange(0, x_axis_size)
        y_ticks = np.arange(0, y_axis_size)
        x_labels = [str(i) for i in x_axis]
        y_labels = [str(i) for i in y_axis][::-1]

        plt.figure(figsize=(10, 6))
        plt.imshow(arr, interpolation='none')
        for i in range(x_axis_size):
            for j in range(y_axis_size):
                plt.text(j, i, f'{(arr[i, j]):.2f}', ha='center', va='center', color='black', size=5)

        plt.xticks(ticks=x_ticks, labels=x_labels, size=6)
        plt.yticks(ticks=y_ticks, labels=y_labels, size=6)
        plt.savefig(fname)
    '''
        
    print('VE predict std:')
    print(np.flipud(ve_predict_std))


    print('Data points amount:')
    print(np.flipud(data_points_amount_map))

    print("VE predicted weighted:")
    print(np.flipud(weighted_ve))

    print("median AFR achieved during RUN:")
    print(np.flipud(AFR_achieved))

    print("median Lambda achieved during RUN:")
    print(np.flipud(Lambda_achieved))

    print("VE increased +:")

    ve_delta = np.flipud(np.round(weighted_ve-VE_TABLE))
    # saveFig(RPM_BINS, KPA_BINS, ve_delta, 'delta.png')

    # saveFig(RPM_BINS, KPA_BINS, RPMdot_achieved, 'rpms.png')

    print(np.flipud(np.round(weighted_ve-VE_TABLE)))

    print("VE increased %:")
    print(np.flipud(weighted_ve/VE_TABLE))

    export(RPM_BINS, KPA_BINS, AFR_TABLE, './output/AFR_EXPORT.table', dtype=float)
    export(RPM_BINS, KPA_BINS, weighted_ve, './output/VE_EXPORT.table')




if __name__ == '__main__':
    main()