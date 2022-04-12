from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon, QPixmap
import sys
sys.path.append('./Src')
from StartUp_GUI import Ui_StartUpWindow 
import os
import qdarkstyle
import yaml
from yaml import Loader, Dumper
import numpy as np
from scipy.optimize import curve_fit
import subprocess
class StartUpApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        self.application=None
        super(StartUpApplicationWindow, self).__init__()
        self.ui = Ui_StartUpWindow()
        self.ui.setupUi(self)
        pixmap = QPixmap('Src\hyspim.png')
        pixmap=pixmap.scaled(400,200)
        self.ui.label.setPixmap(pixmap)
        self.ui.pushButton_recordWindow.clicked.connect(self.showRecordWindow)
        self.ui.pushButton_recordWindowDecoupled.clicked.connect(self.showRecordWindowDecoupled)
        self.ui.pushButton_analyseData.clicked.connect(self.showAnalyzeWindow)
        self.readConfig()
        self.index400, self.index950, self.x_lookup, self.y_lookup=self.lookUpTable()
        print(self.index400, self.index950, self.x_lookup[0], self.y_lookup[0])
        self.offsetX=self.index400
        self.roiX=self.index950-self.index400
        self.writeConfig()

        #lookupFile=open('lookupTable.txt','w')
        #for i in range(len(self.y_lookup)):
        #    lookupFile.write('%.4f\t%.4f\n'%(self.x_lookup[i],self.y_lookup[i]))
        #lookupFile.close()
    def showRecordWindow(self):
        QtWidgets.qApp.processEvents()
        print('t2')
        #os.system('pythonw recordWindow.pyw')
        Add=os.getcwd()+r'\Src'
        subprocess.Popen('pythonw.exe recordWindow.pyw', cwd=Add ,shell=True,)
        print('t3')
        QtWidgets.qApp.processEvents()
    def showRecordWindowDecoupled(self):
        QtWidgets.qApp.processEvents()
        print('t2')
        #os.system('pythonw recordWindow.pyw')
        Add=os.getcwd()+r'\Src'
        subprocess.Popen('pythonw.exe recordWindowDecoupled.pyw', cwd=Add ,shell=True,)
        print('t3')
        QtWidgets.qApp.processEvents()
    def showAnalyzeWindow(self):
        QtWidgets.qApp.processEvents()
        Add=os.getcwd()+r'\Src'
        subprocess.Popen('pythonw.exe analyzeWindow.pyw', cwd=Add,shell=True)
        #os.system('pythonw analyzeWindow.pyw')
        QtWidgets.qApp.processEvents()
    ############################################################
    def writeConfig(self):
        doubleExposureCoeff=int(self.doubleExposureCoeff*1000)
        calibCoeff=np.zeros(shape=(6),dtype=int)
        calibCoeff[0]=int(self.calibCoeff[0]*1000)
        calibCoeff[1]=int(self.calibCoeff[1]*1000)
        calibCoeff[2]=int(self.calibCoeff[2]*1000)
        calibCoeff[3]=int(self.calibCoeff[3]*1000)
        calibCoeff[4]=int(self.calibCoeff[4]*1000)
        calibCoeff[5]=int(self.calibCoeff[5]*1000)
        configGammaMin=int(self.configGammaMin*1000)
        configGammaMax=int(self.configGammaMax*1000)
        nums=[int(self.offsetX),int(self.offsetY),int(self.roiX),int(self.roiY),int(self.binningX),int(self.binningY),int(self.HighRes),doubleExposureCoeff,calibCoeff[0],calibCoeff[1],calibCoeff[2],calibCoeff[3],calibCoeff[4],calibCoeff[5],\
        int(self.offAxisPixNum),int(self.CWRotation),int(self.refFrameNum),int(self.setExposure),int(self.setGain),int(self.setGamma),int(self.scanLen),int(self.maxFrameNum),int(self.flipSecondaryVertical),int(self.flipSecondaryHorizontal),int(self.secondaryVerticalLine),\
        int(self.order),int(self.baslerReverseX),configGammaMin,configGammaMax]
        f=open('Src\config.bin','wb')
        for num in nums:
            #print(format(num,'020b'))
            f.write(format(num,'0100b').encode('ASCII'))
        f.close()
    def readConfig(self):
        f=open('Src\config.bin','r')
        nums=f.read()
        self.calibCoeff=np.zeros(shape=(6),dtype=float)
        self.offsetX=int(nums[0:100],2)
        self.offsetY=int(nums[100:200],2)
        self.roiX=int(nums[200:300],2)
        self.roiY=int(nums[300:400],2)
        self.binningX=int(nums[400:500],2)
        self.binningY=int(nums[500:600],2)
        self.HighRes=bool(int(nums[600:700],2))
        self.doubleExposureCoeff=float(int(nums[700:800],2))/1000.0
        self.calibCoeff[0]=float(int(nums[800:900],2))/1000.0
        self.calibCoeff[1]=float(int(nums[900:1000],2))/1000.0
        self.calibCoeff[2]=float(int(nums[1000:1100],2))/1000.0
        self.calibCoeff[3]=float(int(nums[1100:1200],2))/1000.0
        self.calibCoeff[4]=float(int(nums[1200:1300],2))/1000.0
        self.calibCoeff[5]=float(int(nums[1300:1400],2))/1000.0
        self.offAxisPixNum=int(nums[1400:1500],2)
        self.CWRotation=bool(int(nums[1500:1600],2))
        self.refFrameNum=int(nums[1600:1700],2)
        self.setExposure=int(nums[1700:1800],2)
        self.setGain=int(nums[1800:1900],2)
        self.setGamma=int(nums[1900:2000],2)
        self.scanLen=int(nums[2000:2100],2)
        self.maxFrameNum=int(nums[2100:2200],2)
        self.flipSecondaryVertical=bool(int(nums[2200:2300],2))
        self.flipSecondaryHorizontal=bool(int(nums[2300:2400],2))
        self.secondaryVerticalLine=int(nums[2400:2500],2)
        self.order=int(nums[2500:2600],2)
        self.baslerReverseX=bool(int(nums[2600:2700],2))
        self.configGammaMin=float(int(nums[2700:2800],2))/1000.0
        self.configGammaMax=float(int(nums[2800:2900],2))/1000.0
        f.close()
    ############################################################
    # define the true objective function
    def objective_2(self, x, a, b, c):
        return a*x**2 + b*x + c
    def objective_1(self, x, a, b):
        return a*x + b
    def lookUpTable(self):
        file=open('Src\lookupTable.txt','r')
        x_new=[]
        y_new=[]
        for lines in file:
            x_new.append(float(lines.split()[0]))
            y_new.append(float(lines.split()[1]))
        index400=2*(int(x_new[0]/2))
        index950=x_new[-1]
        for i in range(4):
            if(not (index950-index400)%4==0):
                index950-=1
        file.close()
        return index400, index950, x_new, y_new
    def lookUpTable2(self):
        # load the dataset
        #x_ref=[125.1,223.1,521.5,564.9,663.3,836.5]
        x_ref=self.calibCoeff
        y_ref=[404.7,435.8,532,546.1,578.2,632.8]
        # choose the input and output variables
        # curve fit
        if self.order == 1:
            print('Linear Fitting')
            popt, _ = curve_fit(self.objective_1, x_ref, y_ref)
            a, b = popt
        elif self.order == 2:
            print('Quadratic Fitting')
            popt, _ = curve_fit(self.objective_2, x_ref, y_ref)
            a, b, c = popt
        # summarize the parameter values
        #print('y = %.5f * x^2 + %.5f * x + %.5f' % (a, b, c))
        # plot input vs output
        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.linspace(1,1920,num=1920,endpoint=True,dtype=float)
        # calculate the output for the range
        if self.order == 2 :
                y_line = self.objective_2(x_line, a, b, c)
        elif self.order ==1 :
                y_line = self.objective_1(x_line, a, b)
        index400=list(y_line).index(min(y_line, key=lambda x:abs(x-400)))
        index950=list(y_line).index(min(y_line, key=lambda x:abs(x-950)))
        while((index950-index400)%4!=0):
            if(((index400+1)%2)==0 and (index950-(index400+1))%4==0):
                index400+=1
            elif(((index950+1)%2)==0 and ((index950+1)-index400)%4==0):
                index950+=1
            elif(((index400-1)%2)==0 and (index950-(index400-1))%4==0):
                index400-=1
            elif(((index950-1)%2)==0 and ((index950-1)-index400)%4==0):
                index950-=1
            else:
                index400+=2
        #print(index400,index950)
        #print(y_line[index400],y_line[index950])
        ##########################################################
        #f = interpolate.interp1d(x_line, y_line)
        #x_new = np.linspace(self.offsetX,self.offsetX+self.roiX,num=550,endpoint=True,dtype=float)
        #y_new = f(x_new)   # use interpolation function returned by `interp1d`
        x_new = [sum(x_line[i:i+3])/3 for i in range(index400-1,index950+1,3)]
        y_new = [sum(y_line[i:i+3])/3 for i in range(index400-1,index950+1,3)]
        return index400, index950, x_new, y_new
        # create a line plot for the mapping function
        #for i in range(len(x_line)):
        #	print(i,x_line[i],y_line[i])
        #print('##########################################################')
        #for i in range(len(x_new)):
        #	print(i,x_new[i],y_new[i])
        #pyplot.plot(x_new, y_new, '-', color='black')
        #pyplot.plot(x_ref, y_ref, '*', color='blue')
        #pyplot.plot(x_line, y_line, '--', color='red')
        #pyplot.show()
def StartUpWindow():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    application = StartUpApplicationWindow()
    application.resize(400,400)
    application.show()
    sys.exit(app.exec_())
if __name__ == "__main__":
    StartUpWindow()
