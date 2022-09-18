import numpy as np
np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)
import glob
import pandas as pd
import scipy

from TuneParser import *


#USER settings section

SHIFT_AFR = 1
HITS_NEEDED = 10
QUANTILE = 0.75
MARGIN=1.0



CONFIG_ATMO_6 = {'KPA_MIN':24, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_ATMO_M20 = {'KPA_MIN':28, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_TURBO_M20 = {'KPA_MIN':28, 'KPA_MAX':180, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.0, 'AFR_MAX':12.0}
CONFIG_ATMO_VWKR = {'KPA_MIN':20, 'KPA_MAX':100, 'RPM_MIN':600,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_TURBO_SR20 = {'KPA_MIN':24, 'KPA_MAX':200, 'RPM_MIN':600,'RPM_MAX':7000, 'AFR_MIN':14.5, 'AFR_MAX':11.0}


CURRENT_CONFIG = CONFIG_TURBO_M20


VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
AFR_TABLE_DICT = {'table': 'afrTable', 'xaxis': 'rpmBinsAFR', 'yaxis': 'loadBinsAFR'}
IGN_TABLE_DICT = {'table': 'advTable1', 'xaxis': 'rpmBins2', 'yaxis': 'mapBins1'}
F_NAME = 'CurrentTune.msq'
RPM_BINS_VE, KPA_BINS_VE, _, VE_TABLE_FUNC = getTable(F_NAME,VE_TABLE_DICT)
RPM_BINS_AFR, KPA_BINS_AFR, _, AFR_TABLE_FUNC = getTable(F_NAME,AFR_TABLE_DICT)
RPM_BINS_IGN, KPA_BINS_IGN, _, IGN_TABLE_FUNC = getTable(F_NAME,IGN_TABLE_DICT)


KPA_BINS = ((np.round(np.linspace(CURRENT_CONFIG['KPA_MIN'],CURRENT_CONFIG['KPA_MAX'], 16))//2)*2).astype(int)
RPM_BINS = ((np.round(np.linspace(CURRENT_CONFIG['RPM_MIN'],CURRENT_CONFIG['RPM_MAX'], 16))//100)*100).astype(int)
KPA_MARGIN = ((CURRENT_CONFIG['KPA_MAX'] - CURRENT_CONFIG['KPA_MIN'])//15)*MARGIN
RPM_MARGIN = ((CURRENT_CONFIG['RPM_MAX'] - CURRENT_CONFIG['RPM_MIN'])//15)*MARGIN

print(f"KPA margin: {KPA_MARGIN}")
print(f"RPM margin: {RPM_MARGIN}")
print("bins:")
print(KPA_BINS)
print(RPM_BINS)


VE_TABLE = VE_TABLE_FUNC(RPM_BINS, KPA_BINS)
#IGN_TABLE = IGN_TABLE_FUNC(RPM_BINS, KPA_BINS)
AFR_TABLE = np.array([np.linspace(CURRENT_CONFIG['AFR_MIN'], CURRENT_CONFIG['AFR_MAX'], 16)]*16).transpose()
#AFR_TABLE = AFR_TABLE_FUNC(RPM_BINS, KPA_BINS)
AFR_TABLE_FUNC = scipy.interpolate.interp2d(RPM_BINS, KPA_BINS, AFR_TABLE)






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


flist = glob.glob("./logs/*.msl")




pd_frames = []
for fname in flist:
	fr = pd.read_csv(fname, sep='\t', header=0, skiprows=[0,1,3])
	pd_frames.append(fr)

data_raw = pd.concat(pd_frames)


data_raw.AFR = data_raw.AFR.shift(SHIFT_AFR)
data_raw.Lambda = data_raw.Lambda.shift(SHIFT_AFR)




data = data_raw[(data_raw.Gwarm==100) & (data_raw.RPM>0)& (data_raw.DFCO==0)& (data_raw['rpm/s']>=-1000) & (data_raw['Accel Enrich']==100) & (data_raw.TPS>0.0)]
data = data.dropna()
data['afr_target_func'] = data.apply(lambda row: AFR_TABLE_FUNC(row['RPM'], row['MAP'])[0], axis=1)
data['corr_coef'] = data.apply(lambda row: row['AFR']/row['afr_target_func'], axis=1)
data['ve_predicted'] = data.apply(lambda row: row['VE1']*row['corr_coef'], axis=1)


#prepare a VE bins for statistical work

AFR_achieved =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
Lambda_achieved =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
Lambda_achieved_std =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
VE_achieved = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
pandas_frames = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
VE_predicted = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
VE_predicted_std = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
data_points_amount = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)


for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		kpa_min, kpa_max = np.max([KPA_BINS[i]-KPA_MARGIN, 10]),  np.min([KPA_BINS[i]+KPA_MARGIN, 900])
		rpm_min, rpm_max = np.max([RPM_BINS[j]-RPM_MARGIN,100]),  np.min([RPM_BINS[j]+RPM_MARGIN, 10000])
		pandas_frames[i][j] = data[(data.MAP>=kpa_min) & (data.MAP<=kpa_max) & (data.RPM>=rpm_min) & (data.RPM<=rpm_max)]
		data_points_amount[i][j] = len(pandas_frames[i][j])
		if data_points_amount[i,j]>=HITS_NEEDED:
			AFR_achieved[i][j] = pandas_frames[i][j].AFR.median()
			VE_predicted[i][j]=pandas_frames[i][j]['ve_predicted'].quantile(QUANTILE)
			Lambda_achieved[i][j]=pandas_frames[i][j]['Lambda'].median()


np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)

print("VE opened from tune:")
print(np.flipud(VE_TABLE.astype(np.float64)))


print("Data points amount during RUN:")
print(np.flipud(data_points_amount.astype(int)))


corrected_ve = VE_predicted.astype(np.float64)
corrected_ve = np.nan_to_num(corrected_ve.astype(np.float64))
corrected_ve[corrected_ve==0] = VE_TABLE[corrected_ve==0]

print("VE predicted:")
print(np.flipud((corrected_ve).astype(np.float64)))


print("AFR achieved during RUN:")
print(np.flipud(AFR_achieved.astype(np.float64)))

print("Lambda achieved during RUN:")
print(np.flipud(Lambda_achieved.astype(np.float64)))

print("VE increased +:")
print(np.flipud(np.round(corrected_ve-VE_TABLE)))

print("VE increased %:")
print(np.flipud(corrected_ve/VE_TABLE))




export(RPM_BINS, KPA_BINS, AFR_TABLE, './exports/AFR_EXPORT.table', dtype=float)
export(RPM_BINS, KPA_BINS, corrected_ve, './exports/VE_EXPORT.table')
