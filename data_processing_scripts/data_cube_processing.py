from cProfile import label
from os import read
from re import X
from tkinter import HORIZONTAL
import numpy as np
import matplotlib
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from spectral import *

def readLookUpTable():
    lookupFile=open('Src\\lookupTable.txt','r')
    x_lookUp=[]
    y_lookUp=[]
    for lines in lookupFile:
        x_lookUp.append(float(lines.split()[0]))
        y_lookUp.append(float(lines.split()[1]))
    return y_lookUp


def indexToWaveLen(waveLen):
    look_up_y = readLookUpTable() # wavelength lookup table: data cube x index goes in and wave length is returned
    return look_up_y[waveLen]


# Stores data cube into a 3dimentional array with images being stored on the yz axis and each x layer representing a wavelength
def readDataCube(filePath):
    lines = []
    with open(filePath) as f:
        lines = f.readlines()
        f.close()
        dataCube = []
        for line in lines:
            row = []
            for element in line.split("\t"):
                row.append(float(element))
            dataCube.append(row)

    np_data_cube=np.asarray(dataCube)

    # This loadedArr is a 2D array, therefore
    # we need to convert it to the original
    # array shape.reshaping to get original
    # matrice with original shape.`
    dimA=np.shape(np_data_cube)[0]            #341 or #of wavelength buckets
    dimB=int(np.shape(np_data_cube)[1]/200) # height of scan? <- has to be I think
    dimC=200                                # len of scan

    cube_file=np.zeros(shape=(dimA,dimB,dimC),dtype=int)
    cube_file=np.reshape(np_data_cube,(dimA,dimB,dimC))
    return cube_file

# Writes a 2D array to a file with [brackets] and commas, so it can be
# copied into another python file without loading a matrix
def write2DArrToFile(arr, filePath):
    delimit = ","
    with open(filePath, "a") as f:
        f.write("[\n")
        for index, line in enumerate(arr):
            string = "["
            for index, element in enumerate(line):
                if index == 0:
                    string = string + str(float(element))
                else:
                    string = string + "," + str(float(element))
            if index == len(arr)-1:
                string = string + "]\n"
            else:
                string = string + "],\n"
            f.write(string)
        f.write("]\n")
        f.close()


def swapXZ(mat): # takes in 3d matrix and swaps the x and z indecies
    newMat = []
    for y in range(len(mat[0,:,0])):
        matSlice = mat[:,y,:]
        newMatSlice = np.rot90(matSlice, k=3) # add two more rot 90s to flip the image?
        newMat.append(newMatSlice)

    newMat = np.asarray(newMat)
    return newMat


cube_file = readDataCube('data\\overExposedLeaf.cube')
#cube_file = cube_file[150:300,:,:]

#write2DArrToFile(cube_file[69,:,:], 'data\\layer69.txt')
''' 
labeledLeaf = np.load("data\\labeledLeaf.npy")
labeledLeaf = labeledLeaf.astype(int)

pc = principal_components(cube_file)
xdata = pc.transform(cube_file)
w = view_nd(swapXZ(xdata[:,:,:]), classes=labeledLeaf)
'''

# the view_cube function auto rotates the input 
# data depending on what dimentions are longest 
# slicing to view important parts of cube can fix this
view_cube(swapXZ(cube_file)) 


#Wavelength interpolates between 400 and 950 nm wavelengths
fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [8,1]})
ax1.set_title("Wave Length")
ax1.imshow(cube_file[0,:,:])
#ax1.imshow(pc.reduce(num=10).transform(cube_file))

waveLengthSlider = Slider(ax=ax2, label=f"WaveLength", valmin=0, valmax=(len(cube_file[:,0,0])-1), valinit=0, orientation="horizontal")
def update(val):
    ax1.imshow(cube_file[int(val),:,:])
    ax1.set_title(f"Wave Length: {indexToWaveLen(int(val))}nm")
    fig.canvas.draw_idle()
waveLengthSlider.on_changed(update)

plt.show()

