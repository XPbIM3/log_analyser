import xmltodict
import numpy as np
from scipy import interpolate







class mstable:
    def __init__(self, path):

        with open(path) as fd:
            doc = xmltodict.parse(fd.read())

        zvalues = doc['tableData']['table']['zValues']['#text'].replace('\n','')
        zvalues_str = doc['tableData']['table']['zValues']['#text']

        zvalues = np.fromstring(zvalues, sep=' ').reshape((16,16)).astype(np.float32)


        xaxis = doc['tableData']['table']['xAxis']['#text'].replace(' ','').replace('\n', ' ')
        xaxis = np.fromstring(xaxis, sep=' ').astype(np.uint16)

        yaxis = doc['tableData']['table']['yAxis']['#text'].replace(' ','').replace('\n', ' ')
        yaxis = np.fromstring(yaxis, sep=' ').astype(np.uint16)
        func = interpolate.interp2d(xaxis, yaxis, zvalues)

        self.func = func
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zvalues = zvalues
        self.zvalues_str = zvalues_str

    def arr2str(self, arr):
        return np.array2string(arr).replace(']', '').replace('[','')



ve = mstable('VE.table')
afr = mstable('afr.table')

flatve = ((afr.zvalues/14.7)*ve.zvalues).astype(np.uint8)