from PyQt6.QtWidgets import QMainWindow,QApplication, QWidget, QTableWidget,QTableWidgetItem, QLabel
from PyQt6.QtCore import QMimeData
import sys # Только для доступа к аргументам командной строки
from lib.TuneParser import *
import numpy as np
import matplotlib.pyplot as plt


VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
F_NAME = './input/CurrentTune.msq'



def getBGColor(value, min, max):
    coeff = (value-min)/(max-min)

    r,g,b,a = plt.cm.jet(coeff)
    r=int(r*255)
    g=int(g*255)
    b=int(b*255)

    #b = np.clip(int(((1-coeff)*2)*255), 0,255)
    #r = np.clip(int((coeff*2)*255), 0, 255)
    #g = 0
    return (r,g,b)




class MSTableCellWidget(QLabel):
    def __init__(self, value = np.nan, bgcolor=(255,255,255)):
        assert len(bgcolor)==3
        if value != np.nan:
            super().__init__(str(value))
            self.setStyleSheet(f'background-color: rgb{bgcolor}')

class MSTable(QTableWidget):
    def __init__(self, xaxis, yaxis, table:np.ndarray):
        assert len(xaxis)==16
        assert len(yaxis)==16
        assert table.ndim == 2
        assert table.shape == (16,16)
        super().__init__()
        self.setRowCount(16)
        self.setColumnCount(16)
        table = np.flipud(table)
        self.table = table
        self.min_val = np.min(table)
        self.max_val = np.max(table)

        x_axis = list(map(str, xaxis))
        y_axis = list(map(str, yaxis))[::-1]
        for i in range(16):
            self.setColumnWidth(i,40)
        for i in range(16):
            for j in range(16):
                color = getBGColor(table[i,j], self.min_val, self.max_val)
                self.setCellWidget(i,j, MSTableCellWidget(table[i,j], color))
        self.setHorizontalHeaderLabels(x_axis)
        self.setVerticalHeaderLabels(y_axis)






app = QApplication(sys.argv)
RPM_BINS_VE, KPA_BINS_VE, VE_TABLE, VE_TABLE_FUNC = getTable(F_NAME,VE_TABLE_DICT)
table = MSTable(RPM_BINS_VE, KPA_BINS_VE, VE_TABLE)
table.show()
#app.exec()

