import os
import sys
import numpy as np
from xml_test import mstable

import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

np.set_printoptions(precision = 2, linewidth = 150, suppress = True)

#USER settings section

SHIFT_AFR = 1
DISCARD_BEFORE = 0
DISCARD_AFTER = 5
HITS_NEEDED = 10
T_FULLY_WARMED = 68

STOICH = 14.7

KPA_BINS = np.linspace(10,100,16)
RPM_BINS = np.linspace(600,6000,16)

print(RPM_BINS)
print(KPA_BINS)





def export(RPMS, KPAS, ZS):
	with open('VE.table.template', 'r') as t:
		templ = t.read()

	xax = "\n"+("\n".join(list(map(str, RPMS.astype(np.int)))))+"\n"
	yax = "\n"+("\n".join(list(map(str, KPAS.astype(np.int)))))+"\n"
	zax = "\n"+str(ZS.astype(np.int)).strip('[ ]').replace('[','').replace(']', '')+"\n"


	templ = templ.replace('_XAXIS_', xax)
	templ = templ.replace('_YAXIS_', yax)
	templ = templ.replace('_ZAXIS_', zax)

	with open('VE_export.table', 'w') as t:
		t.write(templ)





def getKpaBin(kpa):
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



def getRpmBin(rpm):
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



FILE = sys.argv[1]

if len(sys.argv)==3:
	PERCENTILE = int(sys.argv[2])
else:
	PERCENTILE = 50

print('PERCENTILE = ', PERCENTILE)



if os.path.exists('VE.table'):
	VE_TABLE_OBJECT = mstable('VE.table')
	grid = np.meshgrid(RPM_BINS, KPA_BINS)
	VE_TABLE = VE_TABLE_OBJECT.func(RPM_BINS, KPA_BINS)


f = open(FILE, 'r')
fWstamp = f.readline().rstrip('\n')
timestamp = f.readline().rstrip('\n')
titles = f.readline().rstrip('\n').split('\t')
units = f.readline().rstrip('\n').split('\t')


rawData = f.readlines()
f.close()
data = []

for rawLine in rawData:
	line = rawLine.rstrip('\n').split('\t')
	if len(line)==len(titles) and 'Time' not in line and 'kpa' not in line:
		data.append(line)

titles = np.array(titles)
AFR = np.where(titles == 'AFR')[0][0]
AFR_TARGET = np.where(titles == 'AFR Target')[0][0]
CLT = np.where(titles == 'CLT')[0][0]
RPM = np.where(titles == 'RPM')[0][0]
ACCENR = np.where(titles == 'Accel Enrich')[0][0]
GWARM = np.where(titles == 'Gwarm')[0][0]
TPSDOT = np.where(titles == 'TPS DOT')[0][0]
DFCO = np.where(titles == 'DFCO')[0][0]
MAP = np.where(titles == 'MAP')[0][0]
CUR_VE = np.where(titles == 'Current VE')[0][0]
RPM_PER_S =np.where(titles == 'rpm/s')[0][0]
TPS = np.where(titles == 'TPS')[0][0]
IAT = np.where(titles == 'IAT')[0][0]
GAMMAE = np.where(titles == 'Gammae')[0][0]

BARO = np.where(titles == 'Baro Pressure')[0][0]
#do a time-shift for AFR readings.
for i in range(len(data)-SHIFT_AFR):
	data[i][AFR] = data[i+SHIFT_AFR][AFR]

for i in range(SHIFT_AFR):
	data.remove(data[-1])





discardFlag = np.zeros(len(data))



usefullData = []

#grab usefull data only

for i, line in enumerate(data):
	#remove anything that:
	#	      coolT 			accelEnrich           WarmEnrich          TPSDot         DFCO                  
	if int(line[CLT])<T_FULLY_WARMED or line[ACCENR]!='100' or line[TPSDOT]!='0' or line[DFCO]!='0' or line[RPM]=='0' or float(line[RPM_PER_S])<0:
		discardFlag[i]=1




#also discard everything near transition points
discardFlagExpanded = discardFlag.copy()
for i in range(DISCARD_BEFORE,len(discardFlag)-DISCARD_AFTER):
	if discardFlag[i]==1:
		for j in range(-DISCARD_BEFORE, +DISCARD_AFTER):
			discardFlagExpanded[i+j]=1




#prepare a VE bins for statistical work

VEBins =  np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)
flatVEBins = np.empty((KPA_BINS.size,RPM_BINS.size), dtype = object)


for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		VEBins[i][j] = []
		flatVEBins[i][j] = []



#fill VE map bins relatively to  load/rpm


for i, line in enumerate(data):
	if discardFlagExpanded[i]!=1:
		load = round(100*(float(line[MAP])/float(line[BARO])))
		rpm = int(line[RPM])

		loadbins = getKpaBin(load)
		rpmbins = getRpmBin(rpm)
		gammae = float(line[GAMMAE])
		
		current_ve = float(line[CUR_VE])
		afr = float(line[AFR])
		afr_target = float(line[AFR_TARGET])

		coef_flat = STOICH/afr
		flat_ve = 100*(current_ve/coef_flat)/gammae

		coef = afr_target/afr
		ve = 100*(current_ve/coef)/gammae


		for l in loadbins:
			for r in rpmbins:
				VEBins[l][r].append(ve)
				flatVEBins[l][r].append(flat_ve)


# aquire median for bins
VEmed = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
flatVEmed = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
std_dev = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)
std_dev_flat = np.zeros((KPA_BINS.size,RPM_BINS.size), dtype = float)


for i in range(KPA_BINS.size):
	for j in range(RPM_BINS.size):
		if (VEBins[i][j]!=[] and len(VEBins[i][j])>HITS_NEEDED):
			VEmed[i][j]=np.percentile(VEBins[i][j],PERCENTILE)
			flatVEmed[i][j]=np.percentile(flatVEBins[i][j],PERCENTILE)
			std_dev[i][j] = np.std(VEBins[i][j])
			std_dev_flat[i][j] = np.std(flatVEBins[i][j])


print('VEs:')
print (np.flipud(VEmed.astype(np.uint8)))


print('flatVEs:')
print (np.flipud(flatVEmed.astype(np.uint8)))


VEmed_filled = VEmed.copy()
VEmed_filled[VEmed_filled==0] = VE_TABLE[VEmed_filled==0]

print('VE filled')
print(VEmed_filled.astype(np.uint8))



#print('std:')
#print (np.flipud(std_dev))
#print('std_flat:')
#print (np.flipud(std_dev_flat))
#RPM_BINS, KPA_BINS = np.meshgrid(RPM_BINS, KPA_BINS)
#surf = ax.plot_surface(RPM_BINS, KPA_BINS, flatVEmed, linewidth=0, antialiased=False)
#plt.show()


export(RPM_BINS, KPA_BINS, VEmed_filled)
