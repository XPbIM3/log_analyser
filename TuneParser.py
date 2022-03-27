import os
import sys
import numpy as np
from scipy import interpolate

def blockToArray(block:list, shape:tuple = (16,16)):
    ret = []
    for b in block:
        ret+= list(map(float, b.strip('\n').strip(' ').split(' ')))
    
    ret = np.array(ret, dtype= float).reshape(shape)
    return ret


def blockToAxis(block):
    return np.array(list(map(float, block)), dtype =int)

def findBlockByName(lines:list, k:str):
    opening_index = -1
    closing_index = -1
    const_str = '\"'
    for i, l in enumerate(lines):
        if l.find(const_str+k+const_str) != -1:
            opening_index=i+1
        
        if opening_index!=-1:
            if l.find("</constant>") != -1:
                closing_index=i
                break
    return(lines[opening_index:closing_index])


VE_TABLE_DICT = {'table': 'veTable', 'xaxis': 'rpmBins', 'yaxis': 'fuelLoadBins'}
AFR_TABLE_DICT = {'table': 'afrTable', 'xaxis': 'rpmBinsAFR', 'yaxis': 'loadBinsAFR'}
F_NAME = 'CurrentTune.msq'




def getTable(path:str, description_dict:dict):
    assert os.path.exists(path)
    with open(path, 'r') as f:
        lines = f.readlines()

    table = findBlockByName(lines, description_dict['table'])
    xaxis = findBlockByName(lines, description_dict['xaxis'])
    yaxis = findBlockByName(lines, description_dict['yaxis'])

    table = blockToArray(table,(16,16))
    xaxis = blockToAxis(xaxis)
    yaxis = blockToAxis(yaxis)
    return table, xaxis, yaxis



vetable = getTable(F_NAME, VE_TABLE_DICT)
afrtable = getTable(F_NAME, AFR_TABLE_DICT)