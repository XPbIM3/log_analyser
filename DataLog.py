import numpy as np
#from xml_test import mstable
import glob
import pandas as pd
#import pywebio
#from pywebio.output import *
import scipy
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

from TuneParser import *
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


#USER settings section

SHIFT_AFR = 1
HITS_NEEDED = 50
T_FULLY_WARMED = 60
STOICH = 14.7
QUANTILE = 0.5
BLURR_FACTOR = 0.5
CONFIG_ATMO_6 = {'KPA_SLOPE':6, 'KPA_OFFSET':10, 'RPM_SLOPE':400,'RPM_OFFSET':600, 'AFR_MIN':15.0, 'AFR_MAX':12.5}
CONFIG_TURBO_6 = {'KPA_SLOPE':12, 'KPA_OFFSET':30, 'RPM_SLOPE':400,'RPM_OFFSET':600, 'AFR_MIN':14.0, 'AFR_MAX':12.0}

CURRENT_CONFIG = CONFIG_ATMO_6
assert CURRENT_CONFIG['KPA_SLOPE']%2 == 0
assert CURRENT_CONFIG['RPM_SLOPE']%100 == 0



#KPA_BINS = np.linspace(25,100,16)
#RPM_BINS = np.linspace(400,7000,16)

KPA_BINS = np.arange(16)*CURRENT_CONFIG['KPA_SLOPE']+CURRENT_CONFIG['KPA_OFFSET']
RPM_BINS = np.arange(16)*CURRENT_CONFIG['RPM_SLOPE']+CURRENT_CONFIG['RPM_OFFSET']


print("defaul bins:")
print(KPA_BINS)
print(RPM_BINS)



VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
AFR_TABLE_DICT = {'table': 'afrTable', 'xaxis': 'rpmBinsAFR', 'yaxis': 'loadBinsAFR'}
F_NAME = 'CurrentTune.msq'
RPM_BINS_VE, KPA_BINS_VE, table, VE_TABLE_FUNC = getTable(F_NAME,VE_TABLE_DICT)
#RPM_BINS_AFR, KPA_BINS_AFR, table, AFR_TABLE_FUNC = getTable(F_NAME,AFR_TABLE_DICT)


VE_TABLE = VE_TABLE_FUNC(RPM_BINS, KPA_BINS)
AFR_TABLE = np.array([np.linspace(CURRENT_CONFIG['AFR_MIN'], CURRENT_CONFIG['AFR_MAX'], 16)]*16).transpose()
#AFR_TABLE = AFR_TABLE_FUNC(RPM_BINS, KPA_BINS)
AFR_TABLE_FUNC = scipy.interpolate.interp2d(RPM_BINS, KPA_BINS, AFR_TABLE)

def pywebioTableRepresentation(table:np.ndarray, xaxis:np.ndarray, yaxis:np.ndarray):
	ret = np.hstack([yaxis.reshape((16,1)), table])
	xaxis_0 = np.hstack([0, xaxis]).reshape((1,17))
	ret = np.vstack([xaxis_0, ret])
	return np.flipud(ret)



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







def getBinFromValue(value:float, bins:np.ndarray):
	return np.argmin(np.abs(value-bins))


def getValueFromBin(val:int, bins:np.ndarray):
	midvalues = []
	for i in range(bins.size-1):
		midvalues.append((bins[i]+bins[i+1])/2)
	midvalues = [0.0] + midvalues + [900.0]

	return (midvalues[val], midvalues[val+1])



#FILE = sys.argv[1]
flist = glob.glob("./logs/*.msl")




pd_frames = []
for fname in flist:
	fr = pd.read_csv(fname, sep='\t', header=0, skiprows=[0,1,3])
	pd_frames.append(fr)

data_raw = pd.concat(pd_frames)

#print(data_raw.describe)

data_raw.AFR = data_raw.AFR.shift(SHIFT_AFR)
data_raw.Lambda = data_raw.Lambda.shift(SHIFT_AFR)




data = data_raw[(data_raw.Gwarm==100) & (data_raw.RPM>0) & (data_raw.DFCO==0) & (data_raw['TPS DOT']==0) & (data_raw['Accel Enrich']==100)]
data = data.dropna()
#print(data.describe)

#data = data.assign(corr_coef = lambda x: x['AFR']/x['AFR Target'])
#data['corr_coef'] = data.apply(lambda row: row['AFR']/row['AFR Target'], axis=1)
data['afr_target_func'] = data.apply(lambda row: AFR_TABLE_FUNC(row['RPM'], row['MAP'])[0], axis=1)
data['corr_coef'] = data.apply(lambda row: row['AFR']/row['afr_target_func'], axis=1)
data['ve_predicted'] = data.apply(lambda row: row['VE1']*row['corr_coef'], axis=1)


#prepare a VE bins for statistical work

AFR_achieved =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
Lambda_achieved =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
Lambda_achieved_std =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
VE_achieved = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
AFR_mismatch = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
pandas_frames = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
VE_predicted = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
VE_predicted_std = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
data_points_amount = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)


for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		kpa_min, kpa_max = getValueFromBin(i, KPA_BINS)
		rpm_min, rpm_max = getValueFromBin(j, RPM_BINS)
		pandas_frames[i][j] = data[(data.MAP>=kpa_min) & (data.MAP<kpa_max) & (data.RPM>=rpm_min) & (data.RPM<rpm_max)]
		data_points_amount[i][j] = len(pandas_frames[i][j])
		AFR_achieved[i][j] = pandas_frames[i][j].AFR.median()
		VE_achieved[i][j]=pandas_frames[i][j].VE1.median()
		VE_predicted[i][j]=pandas_frames[i][j]['ve_predicted'].quantile(QUANTILE)
		VE_predicted_std[i][j]=pandas_frames[i][j]['ve_predicted'].std()
		AFR_mismatch[i][j]=pandas_frames[i][j]['corr_coef'].median()
		Lambda_achieved[i][j]=pandas_frames[i][j]['Lambda'].median()
		Lambda_achieved_std[i][j]=pandas_frames[i][j]['Lambda'].std()


np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)

print("Data points amount:")
print(np.flipud(data_points_amount.astype(int)))


print("VE during RUN:")
print(np.flipud(VE_achieved.astype(np.float64)))

corrected_ve = VE_predicted.astype(np.float64)
corrected_ve = np.nan_to_num(corrected_ve.astype(np.float64))
corrected_ve[corrected_ve==0] = VE_TABLE[corrected_ve==0]

print("VE CORRECTED by coef:")
print(np.flipud((corrected_ve).astype(np.float64)))


blurred_ve = np.round(gaussian_filter(corrected_ve, sigma=BLURR_FACTOR)).astype(int)
blurred_ve[corrected_ve==0] = VE_TABLE[corrected_ve==0]
print("VE blurred:")
print(np.flipud((blurred_ve).astype(np.float64)))




print("predicted VE STD dev:")
print(np.flipud((VE_predicted_std).astype(np.float64)))


print("AFR during RUN:")
print(np.flipud(AFR_achieved.astype(np.float64)))


print("AFR corr coef:")
print(np.flipud(AFR_mismatch.astype(np.float64)))


print("Lambda achieved:")
print(np.flipud(Lambda_achieved.astype(np.float64)))

print("Lambda STD dev:")
print(np.flipud(Lambda_achieved_std.astype(np.float64)))



export(RPM_BINS, KPA_BINS, AFR_TABLE, './exports/AFR_EXPORT.table', dtype=float)
export(RPM_BINS, KPA_BINS, corrected_ve, './exports/VE_EXPORT.table')
export(RPM_BINS, KPA_BINS, blurred_ve, './exports/VE_EXPORT_blurred.table')


