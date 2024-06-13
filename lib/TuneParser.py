import os
import sys
import numpy as np
#import scipy
from scipy.interpolate import RegularGridInterpolator



class tableInterpolate:
    def __init__(self, xaxis: np.ndarray, yaxis: np.ndarray, values:np.ndarray) -> None:
        
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.values = values
        self.obj = RegularGridInterpolator((self.xaxis, self.yaxis), self.values.T, bounds_error=False, fill_value=None)



    def __call__(self, rpm, kpa):
        if np.array(rpm).size == 1 and np.array(kpa).size == 1:
            return self.obj((rpm, kpa))
        else:
            rpms, kpas = np.meshgrid(rpm, kpa)
            return self.obj((rpms, kpas))
        


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
    # obj = tableInterpolate(xaxis, yaxis, table)
    # obj = scipy.interpolate.interp2d(xaxis, yaxis, table)
    obj = tableInterpolate(xaxis, yaxis, table)


    return xaxis, yaxis, table, obj



if __name__ == '__main__':
    np.set_printoptions(floatmode = 'fixed',precision = 2, linewidth = 150, suppress = True)
    x_axis = np.load('xaxis.npy')
    y_axis = np.load('yaxis.npy')
    table = np.load('table.npy')
    obj = tableInterpolate(x_axis, y_axis, table)
    ret = obj(x_axis, y_axis)
    print(ret)
    pass
