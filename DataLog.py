import numpy as np

np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)
import glob
import pandas as pd
#import scipy

from lib.TuneParser import *


#USER settings section

SHIFT_AFR = 1
HITS_NEEDED = 10
QUANTILE = 0.75
MARGIN=1.0



CONFIG_ATMO_6 = {'KPA_MIN':24, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_ATMO_M20 = {'KPA_MIN':28, 'KPA_MAX':100, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_TURBO_M20 = {'KPA_MIN':28, 'KPA_MAX':180, 'RPM_MIN':400,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.0}
CONFIG_ATMO_VWKR = {'KPA_MIN':20, 'KPA_MAX':100, 'RPM_MIN':600,'RPM_MAX':6000, 'AFR_MIN':14.5, 'AFR_MAX':12.5}
CONFIG_TURBO_SR20 = {'KPA_MIN':24, 'KPA_MAX':200, 'RPM_MIN':600,'RPM_MAX':7000, 'AFR_MIN':14.5, 'AFR_MAX':11.0}


CURRENT_CONFIG = CONFIG_TURBO_M20

############################################


VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
AFR_TABLE_DICT = {'table': 'afrTable', 'xaxis': 'rpmBinsAFR', 'yaxis': 'loadBinsAFR'}
IGN_TABLE_DICT = {'table': 'advTable1', 'xaxis': 'rpmBins2', 'yaxis': 'mapBins1'}
F_NAME = './input/CurrentTune.msq'
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


flist = glob.glob("./input/*.msl")




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

AFR_achieved =  np.full((KPA_BINS.size,RPM_BINS.size),np.nan, dtype = float)
Lambda_achieved =  np.full((KPA_BINS.size,RPM_BINS.size),np.nan, dtype = float)
VE_predicted_weightened = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = int)


for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		kpa_min, kpa_max = np.max([KPA_BINS[i]-KPA_MARGIN, 0]),  np.min([KPA_BINS[i]+KPA_MARGIN, 900])
		rpm_min, rpm_max = np.max([RPM_BINS[j]-RPM_MARGIN, 0]),  np.min([RPM_BINS[j]+RPM_MARGIN, 10000])
		pd_local_frame = data[(data.MAP>=kpa_min) & (data.MAP<=kpa_max) & (data.RPM>=rpm_min) & (data.RPM<=rpm_max)]
		data_points_amount = len(pd_local_frame)
		if data_points_amount>=HITS_NEEDED:
			rpms_np = pd_local_frame['RPM'].to_numpy(dtype=int)
			kpas_np = pd_local_frame['MAP'].to_numpy(dtype=int)
			ves_np = pd_local_frame['ve_predicted'].to_numpy(dtype=float)
			x_dist = np.abs(rpms_np - RPM_BINS[j])/RPM_MARGIN
			y_dist = np.abs(kpas_np - KPA_BINS[i])/KPA_MARGIN
			weights = 1-((np.sqrt(x_dist**2 + y_dist**2))/(np.sqrt(2.0)))
			weights = weights/np.sum(weights)
			VE_predicted_weightened[i][j] = int(np.round(np.sum(ves_np * weights)))
			AFR_achieved[i][j] = float(pd_local_frame.AFR.median())
			Lambda_achieved[i][j] = float(pd_local_frame['Lambda'].median())



np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)
weighted_ve = VE_predicted_weightened.astype(float)
weighted_ve[weighted_ve==0] = VE_TABLE[weighted_ve==0]

print("VE predicted weighted:")
print(np.flipud(weighted_ve))

print("median AFR achieved during RUN:")
print(np.flipud(AFR_achieved))

print("median Lambda achieved during RUN:")
print(np.flipud(Lambda_achieved))

print("VE increased +:")
print(np.flipud(np.round(weighted_ve-VE_TABLE)))

print("VE increased %:")
print(np.flipud(weighted_ve/VE_TABLE))

export(RPM_BINS, KPA_BINS, AFR_TABLE, './output/AFR_EXPORT.table', dtype=float)
export(RPM_BINS, KPA_BINS, weighted_ve, './output/VE_EXPORT.table')
