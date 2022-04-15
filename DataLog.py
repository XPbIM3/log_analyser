import numpy as np
#from xml_test import mstable
import glob
import pandas as pd
#import pywebio
import numpy as np
#from pywebio.output import *
import scipy
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

from TuneParser import *
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
np.set_printoptions(precision = 2, linewidth = 150, suppress = True)

#USER settings section

SHIFT_AFR = 1
HITS_NEEDED = 50
T_FULLY_WARMED = 60
STOICH = 14.7
PERCENTILE = 50
BLURR_FACTOR = 0.5

print('PERCENTILE = ', PERCENTILE)

#KPA_BINS = np.linspace(25,100,16)
#RPM_BINS = np.linspace(400,7000,16)

KPA_BINS = np.arange(16)*6+10
RPM_BINS = np.arange(16)*400+600


print("defaul bins:")
print(KPA_BINS)
print(RPM_BINS)



'''
if os.path.exists('VE.table'):
	VE_TABLE_OBJECT = mstable('VE.table')
	#KPA_BINS = np.array(VE_TABLE_OBJECT.yaxis)
	#RPM_BINS = np.array(VE_TABLE_OBJECT.xaxis)
	#print("Bins overrided!")
	#print(KPA_BINS)
	#print(RPM_BINS)
	grid = np.meshgrid(RPM_BINS, KPA_BINS)
	VE_TABLE = VE_TABLE_OBJECT.func(RPM_BINS, KPA_BINS)

if os.path.exists('AFR.table'):
	AFR_TABLE_OBJECT = mstable('AFR.table')
	grid = np.meshgrid(RPM_BINS, KPA_BINS)
	AFR_TABLE = AFR_TABLE_OBJECT.func(RPM_BINS, KPA_BINS)
'''


VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
AFR_TABLE_DICT = {'table': 'afrTable', 'xaxis': 'rpmBinsAFR', 'yaxis': 'loadBinsAFR'}
F_NAME = 'CurrentTune.msq'
RPM_BINS_VE, KPA_BINS_VE, table, VE_TABLE_FUNC = getTable(F_NAME,VE_TABLE_DICT)
#RPM_BINS_AFR, KPA_BINS_AFR, table, AFR_TABLE_FUNC = getTable(F_NAME,AFR_TABLE_DICT)


VE_TABLE = VE_TABLE_FUNC(RPM_BINS, KPA_BINS)
AFR_TABLE = np.array([np.linspace(15.0, 12.5, 16)]*16).transpose()
#AFR_TABLE = AFR_TABLE_FUNC(RPM_BINS, KPA_BINS)
AFR_TABLE_FUNC = scipy.interpolate.interp2d(RPM_BINS, KPA_BINS, AFR_TABLE)

def pywebioTableRepresentation(table:np.ndarray, xaxis:np.ndarray, yaxis:np.ndarray):
	ret = np.hstack([yaxis.reshape((16,1)), table])
	xaxis_0 = np.hstack([0, xaxis]).reshape((1,17))
	ret = np.vstack([xaxis_0, ret])
	return np.flipud(ret)



def export(RPMS, KPAS, ZS, fname):
	with open('VE.table.template', 'r') as t:
		templ = t.read()

	xax = "\n"+("\n".join(list(map(str, RPMS.astype(int)))))+"\n"
	yax = "\n"+("\n".join(list(map(str, KPAS.astype(int)))))+"\n"
	zax = "\n"+str(ZS.astype(int)).strip('[ ]').replace('[','').replace(']', '')+"\n"


	templ = templ.replace('_XAXIS_', xax)
	templ = templ.replace('_YAXIS_', yax)
	templ = templ.replace('_ZAXIS_', zax)

	with open(fname, 'w') as t:
		t.write(templ)



def export_float(RPMS, KPAS, ZS, fname):
	with open('VE.table.template', 'r') as t:
		templ = t.read()

	xax = "\n"+("\n".join(list(map(str, RPMS.astype(int)))))+"\n"
	yax = "\n"+("\n".join(list(map(str, KPAS.astype(int)))))+"\n"
	zax = "\n"+str(ZS.astype(float)).strip('[ ]').replace('[','').replace(']', '')+"\n"


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


'''

def getKpaBin_deprecated(kpa):
	l = KPA_BINS.tolist()
	last_range = KPA_BINS[-1] - KPA_BINS[-2]
	first_range = KPA_BINS[1] - KPA_BINS[0]

	if kpa>=l[-1]  or kpa >= KPA_BINS[-2]+last_range*0.75:
		return [len(l)-1]
	if kpa<=l[0] or kpa< KPA_BINS[1]-first_range*0.75:
		return [0]



	for i in range(1, len(l)-1):
		prev_range = (KPA_BINS[i]-KPA_BINS[i-1])
		next_range = (KPA_BINS[i+1]-KPA_BINS[i])
		
		next_border = KPA_BINS[i]+next_range*0.25
		prev_border = KPA_BINS[i]-prev_range*0.25
		next_border_far = KPA_BINS[i]+next_range*0.75
		prev_border_far = KPA_BINS[i]-prev_range*0.75

		if kpa >= prev_border and kpa <next_border:
			return [i]
		elif kpa < prev_border and kpa >= prev_border_far:
			return [i-1, i]
		elif kpa >= next_border and kpa < next_border_far:
			return [i, i+1]



def getRpmBin_deprecated(rpm):
	l = RPM_BINS.tolist()
	last_range = RPM_BINS[-1] - RPM_BINS[-2]
	first_range = RPM_BINS[1] - RPM_BINS[0]

	if rpm>=l[-1]  or rpm >= RPM_BINS[-2]+last_range*0.75:
		return [len(l)-1]
	if rpm<=l[0] or rpm< RPM_BINS[1]-first_range*0.75:
		return [0]



	for i in range(1, len(l)-1):
		prev_range = (RPM_BINS[i]-RPM_BINS[i-1])
		next_range = (RPM_BINS[i+1]-RPM_BINS[i])
		
		next_border = RPM_BINS[i]+next_range*0.25
		prev_border = RPM_BINS[i]-prev_range*0.25
		next_border_far = RPM_BINS[i]+next_range*0.75
		prev_border_far = RPM_BINS[i]-prev_range*0.75

		if rpm >= prev_border and rpm <next_border:
			return [i]
		elif rpm < prev_border and rpm >= prev_border_far:
			return [i-1, i]
		elif rpm >= next_border and rpm < next_border_far:
			return [i, i+1]


'''
#FILE = sys.argv[1]
flist = glob.glob("./logs/*.msl")




pd_frames = []
for fname in flist:
	fr = pd.read_csv(fname, sep='\t', header=0, skiprows=[0,1,3])
	pd_frames.append(fr)

data_raw = pd.concat(pd_frames)

print(data_raw.describe)

data_raw.AFR = data_raw.AFR.shift(SHIFT_AFR)
data_raw.Lambda = data_raw.Lambda.shift(SHIFT_AFR)


'''
del_indexes = []
if DISCARD_AFTER>0:
	counter = 0
	for index, row in data_raw.iterrows():
		#print(row)
		if row['Accel Enrich']>100:
			counter = DISCARD_AFTER
			del_indexes.append(index)
		elif counter>0:
			row['Accel Enrich']=200
			counter = counter - 1
			del_indexes.append(index)
data_raw.drop(del_indexes, inplace=True)
'''



data = data_raw[(data_raw.Gwarm==100) & (data_raw.RPM>0) & (data_raw.DFCO==0) & (data_raw['TPS DOT']==0) & (data_raw['Accel Enrich']==100)]
data = data.dropna()
print(data.describe)

#data = data.assign(corr_coef = lambda x: x['AFR']/x['AFR Target'])
#data['corr_coef'] = data.apply(lambda row: row['AFR']/row['AFR Target'], axis=1)
data['afr_target_func'] = data.apply(lambda row: AFR_TABLE_FUNC(row['RPM'], row['MAP'])[0], axis=1)
data['corr_coef'] = data.apply(lambda row: row['AFR']/row['afr_target_func'], axis=1)



#prepare a VE bins for statistical work

VEBins =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
error = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
AFR_bins = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
pandas_frames = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)

for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		VEBins[i][j] = []
		error[i][j]=[]
		AFR_bins[i][j]=[]
		kpa_min, kpa_max = getValueFromBin(i, KPA_BINS)
		rpm_min, rpm_max = getValueFromBin(i, RPM_BINS)
		pandas_frames[i][j] = data[(data.MAP>=kpa_min) & (data.MAP<kpa_max) & (data.RPM>=rpm_min) & (data.RPM<rpm_max)]



'''
#fill VE map bins relatively to  load/rpm
iterator = data.iterrows()
for i, line in iterator:
	load = float(line['MAP'])
	rpm = int(line['RPM'])
	loadbins = getKpaBin(load)
	rpmbins = getRpmBin(rpm)
	
	current_ve = float(line['VE1'])
	#afr = float(line['AFR'])
	#afr_target = AFR_TABLE_OBJECT.func(rpm, load)[0]

	#coef = afr_target/afr
	coef = line['corr_coef']
	ve = current_ve * coef
	afr_achieved = line['AFR']

	for l in loadbins:
		for r in rpmbins:
			VEBins[l][r].append(ve)
			error[l][r].append(coef)
			AFR_bins[l][r].append(afr_achieved)


# aquire median for bins
VEmed = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
std_dev = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
error_med = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
binlen = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = int)
AFR_bins_med = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
AFR_bins_std_dev = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		if (VEBins[i][j]!=[] and len(VEBins[i][j])>HITS_NEEDED):
			binlen[i][j] = len(VEBins[i][j])
			VEmed[i][j]=np.percentile(VEBins[i][j],PERCENTILE)
			error_med[i][j]=np.percentile(error[i][j], 50)
			std_dev[i][j] = np.std(VEBins[i][j])
			AFR_bins_med[i][j] = np.percentile(AFR_bins[i][j], 50)
			AFR_bins_std_dev[i][j] = np.std(AFR_bins[i][j])

print("achieved AFR")
print (np.flipud(AFR_bins_med.astype(float)))
#put_text("Achieved AFR during run:")
#put_table(pywebioTableRepresentation(AFR_bins_med, RPM_BINS, KPA_BINS).astype(float).tolist(), header=[])



print('VEs from VE.table:')
print (np.flipud(VE_TABLE.astype(int)))
#put_text("VE Table from tune:")
#put_table(pywebioTableRepresentation(VE_TABLE, RPM_BINS, KPA_BINS).astype(int).tolist(), header=[])

print('VE generated from log(corrected):')
print (np.flipud(np.round(VEmed).astype(int)))
#put_text("VE Table from log, corrected:")
#put_table(pywebioTableRepresentation(np.round(VEmed), RPM_BINS, KPA_BINS).astype(int).tolist(), header=[])


VEmed_filled = np.round(VEmed.copy())
VEmed_filled[VEmed_filled==0] = VE_TABLE[VEmed_filled==0]

print('VE filled')
print(np.flipud(VEmed_filled.astype(int)))
#put_text("VE Table from log, filled up:")
#put_table(pywebioTableRepresentation(VEmed_filled, RPM_BINS, KPA_BINS).astype(int).tolist(), header=[])





print('% more fuel need')
print(np.flipud(error_med))


print('binlen')
print(np.flipud(binlen))

print('std_ve:')
print (np.flipud(std_dev))

print('std_afr:')
print (np.flipud(AFR_bins_std_dev))




#print('std_flat:')
#print (np.flipud(std_dev_flat))
#RPM_BINS, KPA_BINS = np.meshgrid(RPM_BINS, KPA_BINS)
#surf = ax.plot_surface(RPM_BINS, KPA_BINS, flatVEmed, linewidth=0, antialiased=False)
#plt.show()
#pywebio.session.set_env(output_max_width = '100%')



blurred_ve = np.round(gaussian_filter(VEmed_filled, sigma=BLURR_FACTOR)).astype(int)

export_float(RPM_BINS, KPA_BINS, AFR_TABLE, 'AFR_EXPORT.table')
export(RPM_BINS, KPA_BINS, VEmed_filled, 'VE_EXPORT.table')
export(RPM_BINS, KPA_BINS, blurred_ve, 'VE_EXPORT_blurred.table')

'''