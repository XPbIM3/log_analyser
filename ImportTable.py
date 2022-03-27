import numpy as np
import scipy
from scipy import interpolate
	

class mstable:
	def __init__(self, PATH=None, title=None):
		self.data = np.zeros((16,16), dtype=np.float)
		self.xaxis = np.zeros(16,dtype=np.float)
		self.xaxis_name='RPM'
		self.yaxis = np.zeros(16,dtype=np.float)
		self.yaxis_name='LOAD'
		#self.func = object



		if PATH!=None and type(PATH)==str:
			with open(PATH) as f:
				RPM = np.zeros(16, dtype=np.float)
				LOAD = np.zeros(16, dtype=np.float)
				DATA = np.zeros((16,16), dtype=np.float)
				f = open(PATH)
				while f.readline().find('<xAxis')==-1:
					pass
				for i in range(16):
					RPM[i]=f.readline()
				while f.readline().find('<yAxis')==-1:
					pass
				for i in range(16):
					LOAD[i]=f.readline()
				while f.readline().find('<zValues')==-1:
					pass
				for i in range(16):
					DATA[i]=np.array(f.readline().strip('\n').strip(' ').split(' '), dtype=np.float)
			self.xaxis = RPM
			self.yaxis = LOAD
			self.data = DATA
			self.func = scipy.interpolate.interp2d(self.xaxis, self.yaxis, self.data)


	def export_text(self):
		return (' '+ str(np.int8(self.data)).replace(']','').replace('[',''))

	def __repr__(self):
		rep = np.zeros((17,17), dtype = np.float)
		rep[1:17,0]=self.yaxis
		rep[0,1:17]=self.xaxis
		rep[1:17, 1:17] = self.data
		rep[0,0] = -1
		return (str(rep[::-1,::]))


