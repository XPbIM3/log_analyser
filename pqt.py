from PyQt6.QtWidgets import QMainWindow,QApplication, QWidget, QTableWidget,QTableWidgetItem, QLabel, QToolTip
from PyQt6 import QtCore
from PyQt6 import QtGui

import cv2 as cv
from lib.TuneParser import *
import numpy as np

import matplotlib.pyplot as plt
from uuid import uuid1

VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
F_NAME = './input/CurrentTune.msq'



def getBGColor(value, min, max):
    coeff = (value-min)/(max-min)
    r,g,b,a = plt.cm.jet(coeff)
    r=int(r*255)
    g=int(g*255)
    b=int(b*255)
    a = 150
    #b = np.clip(int(((1-coeff)*2)*255), 0,255)
    #r = np.clip(int((coeff*2)*255), 0, 255)
    #g = 0
    return (r,g,b,a)






class MSTableCellWidgetWHistogramm(QLabel):
    def __init__(self, value:np.ndarray,  bgcolor:tuple, hist_data:np.ndarray):
        uid = uuid1()
        path = f'tmp/{uid}.svg'
        plt.clf()
        plt.hist(hist_data)
        plt.savefig(path)
        plt.clf()
        assert len(bgcolor)==4
        if value != np.nan:
            super().__init__(str(int(np.round(value))))
            self.setStyleSheet(f'background: rgba{bgcolor}')
            #self.setStyleSheet(f'background: rgba(255,255,255,0)')
        else:
            super().__init__()
        
        self.setToolTip(f'<img src="{path}">')



class MSTableCellWidget(QLabel):
    def __init__(self, value = np.nan,bgcolor=(255,255,255,0)):
        assert len(bgcolor)==4
        if value != np.nan:
            super().__init__(str(int(np.round(value))))
            self.setStyleSheet(f'background: rgba{bgcolor}')
            #self.setStyleSheet(f'background: rgba(255,255,255,0)')
        else:
            super().__init__("")
      






        

class MSTable(QTableWidget):
    def __init__(self, xaxis, yaxis, table:np.ndarray):
        assert table.ndim == 2
        super().__init__()
        self.y_shape = len(yaxis)
        self.x_shape = len(xaxis)
        assert table.shape == (self.y_shape,self.x_shape)

        self.setRowCount(self.x_shape)
        self.setColumnCount(self.y_shape)
        table = np.flipud(table)
        self.table = table
        self.min_val = np.min(table)
        self.max_val = np.max(table)
        x_axis = list(map(str, xaxis))
        y_axis = list(map(str, yaxis))[::-1]
        for i in range(self.x_shape):
            self.setColumnWidth(i,20)
        for i in range(self.x_shape):
            for j in range(self.y_shape):
                color = getBGColor(table[i,j], self.min_val, self.max_val)
                #self.setCellWidget(i,j, MSTableCellWidget(table[i,j], color))
                self.setCellWidget(i,j, MSTableCellWidgetWHistogramm(table[i,j], color, np.random.randint(10,20,100)))
                widget = self.cellWidget(i,j)
                #widget.setToolTip('asd')

        self.setHorizontalHeaderLabels(x_axis)
        self.setVerticalHeaderLabels(y_axis)






app = QApplication([])
RPM_BINS_VE, KPA_BINS_VE, VE_TABLE, VE_TABLE_FUNC = getTable(F_NAME,VE_TABLE_DICT)
table = MSTable(RPM_BINS_VE, KPA_BINS_VE, VE_TABLE)
table.show()


app.exec()

