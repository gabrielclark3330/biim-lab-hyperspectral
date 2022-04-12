from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from AnalyzeWindowAutoLoad_GUI import  Ui_AnalyzeWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QRubberBand, QWidget
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import os
import qdarkstyle
import sys
from glob import glob
import pickle
import numpy as np
import cv2
import numpy as np
import scipy.ndimage
import pickle
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot
from scipy import interpolate
import yaml
from yaml import Loader, Dumper
from PIL import Image
############################################################
class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(111)
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()
############################################################
global WriteAddress
WriteAddress='.temp'
class LoadingWindow(QtWidgets.QMessageBox):
    def __init__(self):
        super(LoadingWindow, self).__init__()
        self.setWindowTitle("  ")
        self.setText("Please Wait\n\n\nSaving Data.\n")
        self.setIcon(QMessageBox.Information)
    def closeEvent(self, event):
        event.accept()
class AnalyzeApplicationWindow(QtWidgets.QMainWindow):
    resized = QtCore.pyqtSignal()
    def __init__(self):
        #self.application=None
        super(AnalyzeApplicationWindow, self).__init__()
        self.ui = Ui_AnalyzeWindow()
        self.ui.setupUi(self)
        self.readConfig()
        self.ui.pushButton_saveCube.clicked.connect(self.saveCube)
        self.ui.pushButton_RGBStat.clicked.connect(self.changeRGBStat)
        self.cubeSaved=False
    def readConfig(self):
        f=open('config.bin','r')
        nums=f.read()
        self.offsetX=int(nums[0:100],2)
        self.offsetY=int(nums[100:200],2)
        self.roiX=int(nums[200:300],2)
        self.roiY=int(nums[300:400],2)
        self.binningX=int(nums[400:500],2)
        self.binningY=int(nums[500:600],2)
        self.HighRes=bool(int(nums[600:700],2))
        self.doubleExposureCoeff=float(int(nums[700:800],2))/1000.0
        self.offAxisPixNum=int(nums[800:900],2)
        self.CWRotation=bool(int(nums[900:1000],2))
        self.refFrameNum=int(nums[1000:1100],2)
        self.setExposure=int(nums[1100:1200],2)
        self.setGain=int(nums[1200:1300],2)
        self.setGamma=int(nums[1300:1400],2)
        self.scanLen=int(nums[1400:1500],2)
        self.maxFrameNum=int(nums[1500:1600],2)
        self.flipSecondaryVertical=bool(int(nums[1600:1700],2))
        self.flipSecondaryHorizontal=bool(int(nums[1700:1800],2))
        self.secondaryVerticalLine=int(nums[1800:1900],2)
        self.order=int(nums[1900:2000],2)
        self.baslerReverseX=bool(int(nums[2000:2100],2))
        self.configGammaMin=float(int(nums[2100:2200],2))/1000.0
        self.configGammaMax=float(int(nums[2200:2300],2))/1000.0
        self.calibCoeffNum=int(nums[2300:2400],2)
        self.calibCoeff_x=np.zeros(shape=(self.calibCoeffNum),dtype=float)
        self.calibCoeff_y=np.zeros(shape=(self.calibCoeffNum),dtype=float)
        cx=0
        cy=0
        for i in range(2*self.calibCoeffNum):
            if(i%2==0):
                self.calibCoeff_x[cx]=float(int(nums[2400+(i*100):2400+((i+1)*100)],2))/1000.0
                cx+=1
            else:
                self.calibCoeff_y[cy]=float(int(nums[2400+(i*100):2400+((i+1)*100)],2))/1000.0
                cy+=1
        f.close()
        ########################################################################
    def saveCube(self):
        try:
            self.CubeFile
        except AttributeError:
            return
        #print('saving')
        dimA=np.shape(self.CubeFile)[0]
        dimB=np.shape(self.CubeFile)[1]
        dimC=np.shape(self.CubeFile)[2]
        arr_reshaped = np.reshape(self.CubeFile,(dimA,dimB*dimC))
        fname , _ = QFileDialog.getSaveFileName(self, 'Save file', '.',"Cube file (*.cube)")
        if (len(fname)>0):    
            QtWidgets.qApp.processEvents()
            msg=LoadingWindow()
            msg.show()
            QtWidgets.qApp.processEvents()
            np.savetxt(fname, arr_reshaped, fmt='%.4f', delimiter='\t')
            self.cubeSaved=True
            sys.exit()
    def readLookUpTable(self):
        lookupFile=open('lookupTable.txt','r')
        self.x_lookUp=[]
        self.y_lookUp=[]
        for lines in lookupFile:
            self.x_lookUp.append(float(lines.split()[0]))
            self.y_lookUp.append(float(lines.split()[1]))
    def run(self):
        self.ui.pushButton_RGBStat.setEnabled(False)
        self.RGBStat=False
        self.readLookUpTable()
        self.im_depth=self.ui.horizontalSlider.value()
        self.ui.horizontalSlider.valueChanged.connect(self.update_im_depth)
        self.resized.connect(self.someFunction)
        self.loadData()
        #for i in range(len(x_lookup)):
        #    print(i,self.x_lookup[i],self.y_lookUp[i])
############################################################
        self.canvas = MplCanvas(self, width=5, height=2, dpi=100)
        self.canvas.axes.set_ylim(0,1.1)
        self.canvas.axes.set_xlim(400,950)
        self.canvas.axes.set_xlabel('(nm)',color='white')#,labelpad=-10)
        self.major_ticks_x = np.arange(400,1000, 50)
        self.canvas.axes.set_xticks(self.major_ticks_x)
        self.canvas.axes.grid(which='major',color='red' ,alpha=0.5)
        self.minor_ticks_x = np.arange(400,950, 10)
        self.canvas.axes.set_xticks(self.minor_ticks_x, minor=True)
        self.canvas.axes.grid(which='minor',color='red', alpha=0.2)
        self.canvas.axes.yaxis.grid(True,linestyle='--',color='red')
        self.canvas.axes.tick_params(axis='both',labelcolor="white",color="white")
        self.canvas.axes.spines['top'].set_color("white")
        self.canvas.axes.spines['left'].set_color("white")
        self.canvas.axes.spines['right'].set_color("white")
        self.canvas.axes.spines['bottom'].set_color("white")
        self.canvas.axes.set_facecolor((25.0/255.0,35.0/255.0,45.0/255.0))
        self.canvas.figure.set_facecolor((25.0/255.0,35.0/255.0,45.0/255.0))
        self.ui.gridLayout.addWidget(self.canvas, 4,0,1,6)
        self.line=None
############################################################
        #start rubberband
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        bla = QtGui.QPalette()
        #bla.setBrush(QtGui.QPalette.Highlight, QtGui.QBrush())#QtCore.Qt.redi))
        bla.setBrush(QtGui.QPalette.Text, QtGui.QBrush(QtCore.Qt.red))
        self.rubberband.setPalette(bla)
        self.rubberband.setWindowOpacity(100.0)
    def changeRGBStat(self):
        if self.RGBStat:
            self.RGBStat=False
            self.ui.pushButton_RGBStat.setText('RGB')
            self.ui.horizontalSlider.setEnabled(True)
        else:
            self.RGBStat=True
            self.ui.pushButton_RGBStat.setText('HSI')
            self.ui.horizontalSlider.setEnabled(False)
        self.basler_update_image()
    def mousePressEvent(self, event):
        try:
            self.rubberband
        except AttributeError:
            return
        except NameError:
            return
        if(event.x()<self.ui.label_Image.x() or event.x()>self.ui.label_Image.x()+self.ui.label_Image.width()):
            return
        if(event.y()<self.ui.label_Image.y() or event.y()>self.ui.label_Image.y()+self.ui.label_Image.height()):
            return
        self.origin = event.pos()#QtCore.QPoint(self.ui.label_Image.x(),self.ui.label_Image.y())
        #self.rubberband.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
        x=self.origin.x()
        y=self.origin.y()
        self.rubberband.setGeometry(x,y,4,4)#x+2,y,y+2)
        self.rubberband.show()
        QWidget.mousePressEvent(self, event)
    def mouseMoveEvent(self, event):
        try:
            self.rubberband
        except AttributeError:
            return
        except NameError:
            return
        if(event.x()<self.ui.label_Image.x() or event.x()>self.ui.label_Image.x()+self.ui.label_Image.width()):
            return
        if(event.y()<self.ui.label_Image.y() or event.y()>self.ui.label_Image.y()+self.ui.label_Image.height()):
            return
        if self.rubberband.isVisible():
            self.rubberband.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())
        QWidget.mouseMoveEvent(self, event)
    def mouseReleaseEvent(self, event):
        try:
            self.rubberband
        except AttributeError:
            return
        except NameError:
            return
        if self.rubberband.isVisible():
            self.selRect = self.rubberband.geometry()
            #self.rubberband.hide()
            self.SelectedX=int((self.selRect.x()-self.ui.label_Image.x())/self.scaleX)
            self.SelectedY=int((self.selRect.y()-self.ui.label_Image.y())/self.scaleY)
            self.SelectedWidth=int(self.selRect.width()/self.scaleX)
            self.SelectedHeight=int(self.selRect.height()/self.scaleY)
            #print('RubberBand: x,width,y,height')
            #print(self.SelectedX,self.SelectedWidth,self.SelectedY,self.SelectedHeight)
        if(True):
            try:
                self.SelectedHeight
                self.SelectedWidth
                self.SelectedHeight=max(self.SelectedHeight,1)
                self.SelectedWidth=max(self.SelectedWidth,1)
            except AttributeError:
                return
            except NameError:
                return
            len=np.shape(self.CubeFile)[0]-1
            list2=np.zeros(shape=(len))
            if(self.SelectedHeight>0 or self.SelectedWidth>0):
                for i in range(len):
                    list2[i]=np.sum(self.CubeFile[i,self.SelectedY:self.SelectedY+self.SelectedHeight,self.SelectedX:self.SelectedX+self.SelectedWidth])
                list2[:]/=self.SelectedHeight*self.SelectedWidth
            else:
                list2=self.CubeFile[:,self.SelectedY,self.SelectedX]
            self.canvas.axes.set_facecolor((25.0/255.0,35.0/255.0,45.0/255.0))
            self.canvas.axes.set_ylim(0,1.1)
            self.canvas.axes.set_xlim(400,950)
            self.canvas.axes.set_xlabel('(nm)',color='white')#,labelpad=-10)
            self.major_ticks_x = np.arange(400,1000, 50)
            self.canvas.axes.set_xticks(self.major_ticks_x)
            self.canvas.axes.grid(which='major',color='red' ,alpha=0.5)
            self.minor_ticks_x = np.arange(400,950, 10)
            self.canvas.axes.set_xticks(self.minor_ticks_x, minor=True)
            self.canvas.axes.grid(which='minor',color='red', alpha=0.2)
            if self.line is None:
                plot_ref=self.canvas.axes.plot(self.y_lookUp,list2,color='yellow')
                self.line=plot_ref[0]
            else:
                self.line.set_ydata(list2)
            self.canvas.axes.yaxis.grid(True,linestyle='--')
            self.canvas.draw()
        else:
            self.canvas.axes.clear()
            #self.canvas.axes.cla()
            self.line=None
        QWidget.mouseReleaseEvent(self, event)
############################################################
    def resizeEvent(self, event):
        self.resized.emit()
        return super(AnalyzeApplicationWindow, self).resizeEvent(event)
    def someFunction(self):
        self.dataDisplayWidth =self.ui.label_Image.width()
        self.dataDisplayHeight = self.ui.label_Image.height()
        self.update_im_depth()
############################################################
    def loadData(self):
        global WriteAddress
        fileList=glob(os.path.join(WriteAddress,'out*.pick'))
        #print('FileList:')
        #print(fileList)
        ob1=[]
        for file in fileList:
            ff = open(file,"rb")
            tempFile=pickle.load(ff)
            ob1.extend(tempFile[:])
            ff.close()
        #object_files=self.Spectral_Interpolate(ob1)
        #f_ref=open(os.path.join(WriteAddress,'reference.pick'),'rb')
        try:
            f_ref=open('./reference.pick','rb')
            refFrames=pickle.load(f_ref)
            for i in range(self.refFrameNum):
                if i == 0:
                    refImg=refFrames[i].astype(int)
                else:
                    refImg+=refFrames[i].astype(int)
            refImg=(refImg/self.refFrameNum).astype(np.uint16)
            self.withRef=True
            refImg[refImg==0]=1
        except FileNotFoundError:
            self.withRef=False

        len_ob1=len(ob1)
        maxValob1=2**16-1
        for i in range(len_ob1):
            percentage=(100.0*i)/len_ob1
            str_per='Setting Reference: %.2f'%percentage
            QtWidgets.qApp.processEvents()
            self.ui.progressBar.setValue(int(percentage))
            self.ui.progressBar.setFormat('Setting Reference: %.1f %%'%percentage)
            self.ui.label_Image.setText(str_per)
            QtWidgets.qApp.processEvents()
            if self.withRef:
                #('shape',np.shape(ob1[i]),np.shape(refImg))
                #print('befor',i,ob1[i][100][100:110])
                #print('refer',i,refImg[100][100:110])
                ob1[i]=np.divide(ob1[i],refImg)
                #print('after',i,ob1[i][100][100:110])
                #print('locat',i,np.max(ob1[i]),np.min(ob1[i]))
            else:
                ob1[i]=np.divide(ob1[i],maxValob1)
        ob2=self.Spectral_Interpolate(ob1)
        #print('Spectral',np.max(ob2),np.min(ob2))
        object_files =self.Rzx_Interpolate(ob2)
        #print('RZX',np.max(object_files),np.min(object_files))
        object_files=np.clip(object_files,0,1)
        #print('clip',np.max(object_files),np.min(object_files))
        dimA=len(object_files)
        dimB=len(object_files[0])
        dimC=len(object_files[0][0])
        self.imNum=dimC
        self.ui.horizontalSlider.setMaximum(dimC-2)
        self.ui.horizontalSlider.setValue(int((dimC-2)/2))
        self.CubeFile=np.zeros(shape=(dimC,dimB,self.maxFrameNum),dtype=float)
        self.CubeFile_2=np.zeros(shape=(dimC,dimB,dimA),dtype=float)
        self.CubeFile_2=np.swapaxes(object_files,2,0)
        self.CubeFile[0:dimC-1,0:dimB-1,0:dimA-1]=self.CubeFile_2[0:dimC-1,0:dimB-1,0:dimA-1]
        #print(np.shape(self.CubeFile))
        #print(np.shape(self.CubeFile_2))
        self.dataDisplayWidth=self.ui.label_Image.width()
        self.dataDisplayHeight=self.ui.label_Image.height()
        self.basler_update_image()
############################################################
    def Spectral_Interpolate(self,inp_image):
        ROI_X=0
        ROI_Y=0
        ROI_Width=self.roiX 
        ROI_Height=self.roiY
        if self.HighRes:
            ROI_CROP=self.roiY
        else:
            ROI_CROP=100
        out_image=[]
        len_inp_image=len(inp_image)
        self.ui.progressBar.setVisible(True)
        for i in range(len_inp_image):
            #print(i)
            percentage=(100.0*i)/len_inp_image
            str_per='LOADING Images: %.2f'%percentage
            QtWidgets.qApp.processEvents()
            self.ui.progressBar.setValue(int(percentage))
            self.ui.progressBar.setFormat('Loading Images: %.1f %%'%percentage)
            self.ui.label_Image.setText(str_per)
            QtWidgets.qApp.processEvents()
            ROI_Img=inp_image[i][0:ROI_Height][0:ROI_Width]
            out_image.append(cv2.resize(ROI_Img,None, fx = (len(self.y_lookUp)+1)/self.roiX, fy = ROI_CROP/self.roiY, interpolation = cv2.INTER_CUBIC))
        self.ui.progressBar.setVisible(False)
        return out_image
    def Rzx_Interpolate(self,inp_image):
        if self.offAxisPixNum==0:
            return inp_image
        ROI_X=0
        ROI_Y=0
        ROI_Width=(len(self.y_lookUp)+1) 
        if self.HighRes:
            ROI_CROP=self.roiY
        else:
            ROI_CROP=100
        ROI_Height=ROI_CROP
        outOfPlane=self.offAxisPixNum*ROI_CROP/self.roiY
        #out_image=np.zeros(shape=(len(inp_image),len(inp_image[0]),len(inp_image[0][0])))
        out_image=[]
        len_inp_image=len(inp_image)
        self.ui.progressBar.setVisible(True)
        for i in range(len_inp_image):
            self.update()
            #print(i)
            percentage=(100.0*i)/len_inp_image
            str_per='Processing: %.1f %%'%percentage
            QtWidgets.qApp.processEvents()
            self.ui.progressBar.setValue(int(percentage))
            self.ui.progressBar.setFormat('Rotating: %.1f %%'%percentage)
            self.ui.label_Image.setText(str_per)
            QtWidgets.qApp.processEvents()
            ROI_Img=inp_image[i][ROI_Y:ROI_Y+ROI_Height][ROI_X:ROI_X+ROI_Width]
            bicubic_img = cv2.resize(ROI_Img,None, fx = 1, fy = (ROI_Width+0.0)/outOfPlane, interpolation = cv2.INTER_CUBIC)
            #print(np.shape(ROI_Img),np.shape(bicubic_img))
            mv_img=bicubic_img.copy()
            mv_img[:,:]=0
            IntImHeight, IntImWidth = bicubic_img.shape
            for i in range(IntImWidth):
                if not self.CWRotation:
                    mv_img[0:IntImHeight-i,i]=bicubic_img[i:IntImHeight,i]
                else : 
                    mv_img[0:IntImHeight-IntImWidth+i,i]=bicubic_img[IntImWidth-i:IntImHeight,i]
            mv_img_2=mv_img[0:(IntImHeight-IntImWidth),:]
            out_image.append(cv2.resize(mv_img_2,None, fx = 1, fy = outOfPlane/ROI_Width, interpolation = cv2.INTER_CUBIC))
        self.ui.progressBar.setVisible(False)
        return out_image
############################################################
    def update_im_depth(self):
        self.im_depth=self.ui.horizontalSlider.value()
        text='Wave Length: %.1f nm'%self.y_lookUp[self.im_depth]
        self.ui.label_waveLength.setText(text)
        self.basler_update_image()
    def basler_update_image(self):
        #qt_img = (self.CubeFile[self.im_depth])
        #cv2.imshow('im',self.CubeFile[self.im_depth])
        try:
            self.CubeFile
        except AttributeError:
            return
        qt_img = self.convert_cv_qt_basler(self.CubeFile[self.im_depth],not self.RGBStat)
        self.ui.label_Image.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored))
        self.ui.label_Image.setPixmap(qt_img)
    def convert_cv_qt_basler(self, basler_cv_img,stat):
        """Convert from an opencv image to QPixmap"""
        #print(basler_cv_img*256)
        #print(np.max(basler_cv_img))
        #print(np.min(basler_cv_img))
        if stat:
            bb=(basler_cv_img*255).astype(np.uint8)
            convert_to_Qt_format = QtGui.QImage(bb.tobytes(), bb.shape[1], bb.shape[0], bb.shape[1], QtGui.QImage.Format_Grayscale8)
            self.scaleX=self.ui.label_Image.width()/(bb.shape[1]+0.0)
            self.scaleY=self.ui.label_Image.height()/(bb.shape[0]+0.0)
            p = convert_to_Qt_format.scaled(self.ui.label_Image.width(),self.ui.label_Image.height())#, Qt.KeepAspectRatio)
            return QtGui.QPixmap.fromImage(p)
        else:
            xyzbar=[0.0143100000000000,0.000396000000000000,0.0678500000000000,
                    0.0435100000000000,0.00121000000000000, 0.207400000000000,
                    0.134380000000000, 0.00400000000000000, 0.645600000000000,
                    0.283900000000000,0.0116000000000000,
                    1.38560000000000,0.348280000000000,0.0230000000000000,1.74706000000000,
                    0.336200000000000,0.0380000000000000,1.77211000000000,0.290800000000000,
                    0.0600000000000000,1.66920000000000,0.195360000000000,0.0909800000000000,
                    1.28764000000000,0.0956400000000000,0.139020000000000,0.812950000000000,
                    0.0320100000000000,0.208020000000000,0.465180000000000,0.00490000000000000,
                    0.323000000000000,0.272000000000000,0.00930000000000000,0.503000000000000,
                    0.158200000000000,0.0632700000000000,0.710000000000000,0.0782500000000000,
                    0.165500000000000,0.862000000000000,0.0421600000000000,0.290400000000000,
                    0.954000000000000,0.0203000000000000,0.433450000000000,0.994950000000000,
                    0.00875000000000000,0.594500000000000,0.995000000000000,0.00390000000000000,
                    0.762100000000000,0.952000000000000,0.00210000000000000,0.916300000000000,
                    0.870000000000000,0.00165000000000000,1.02630000000000,0.757000000000000,
                    0.00110000000000000,1.06220000000000,0.631000000000000,0.000800000000000000,
                    1.00260000000000,0.503000000000000,0.000340000000000000,0.854450000000000,
                    0.381000000000000,0.000190000000000000,0.642400000000000,0.265000000000000,
                    5.00000000000000e-05,0.447900000000000,0.175000000000000,2.00000000000000e-05,
                    0.283500000000000,0.107000000000000,0,0.164900000000000,0.0610000000000000,0,
                    0.0874000000000000,0.0320000000000000,0,0.0467700000000000,0.0170000000000000,
                    0,0.0227000000000000,0.00821000000000000,0,0.0113590000000000,0.00410200000000000,
                    0,0.00579000000000000,0.00209100000000000,0,0.00289900000000000,0.00104700000000000,0]
            r=xyzbar[0::3]
            g=xyzbar[1::3]
            b=xyzbar[2::3]
            x=np.linspace(0,self.imNum,num=len(r))
            interp3R = interpolate.interp1d(x, r, kind = "cubic")
            interp3G = interpolate.interp1d(x, g, kind = "cubic")
            interp3B = interpolate.interp1d(x, b, kind = "cubic")
            x_new=np.linspace(0,self.imNum,num=self.imNum)
            r_new=interp3R(x_new)
            g_new=interp3G(x_new)
            b_new=interp3B(x_new)
            #plt.plot(x_new,r_new)
            #plt.plot(x,r)
            #plt.plot(x_new,g_new)
            #plt.plot(x,g)
            #plt.plot(x_new,b_new)
            #plt.plot(x,b)
            #plt.show()
            #rgb_array = cv2.imread('Ok.png')
            #rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
            lenn,width,height=np.shape(self.CubeFile)
            rgb_array=np.zeros(shape=(width,height,3),dtype=float) 
            for i in range(lenn):
                rgb_array[:,:,0]+=self.CubeFile[i,:,:]*r_new[i]
                rgb_array[:,:,1]+=self.CubeFile[i,:,:]*g_new[i]
                rgb_array[:,:,2]+=self.CubeFile[i,:,:]*b_new[i]
            max_r=np.max(rgb_array)
            max_g=np.max(rgb_array)
            max_b=np.max(rgb_array)
            rgb_array[:,:,0]=rgb_array[:,:,0]*(255.0/max_r)
            rgb_array[:,:,1]=rgb_array[:,:,1]*(255.0/max_g)
            rgb_array[:,:,2]=rgb_array[:,:,2]*(255.0/max_b)
            #rgb_array=Image.fromarray(rgb_array)
            h, w, ch = np.shape(rgb_array)
            bytesPerLine = ch * w
            qImg = QImage(rgb_array.astype(np.uint8), w, h, bytesPerLine, QImage.Format_RGB888)
            p = qImg.scaled(self.ui.label_Image.width(),self.ui.label_Image.height())#, Qt.KeepAspectRatio)
            return(QPixmap.fromImage(p))


############################################################
    def saveMat(self):
        global WriteAddress
        os.chdir(WriteAddress)
        mdic = {"Data": CubeFile.tolist(), "label": "experiment"}
        savemat(filename, mdic)
        self.baslerThread.dd=[]
############################################################
    def closeEvent(self, event):
        if not self.cubeSaved:
            qm = QMessageBox()
            qm.setIcon(QMessageBox.Warning)
            ret = qm.question(self,'', "Cube File Not Saved\nContinue?", qm.Yes | qm.No)
            if (ret == qm.Yes):
                event.accept()
            elif(ret==qm.No):
                event.ignore()
        else:
            event.accept()
import time
def AnalyzeWindow():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    application = AnalyzeApplicationWindow()
    application.resize(800,800)
    application.ui.label_Image.setText(str(time.time()))
    application.ui.label_Image.setAlignment(QtCore.Qt.AlignCenter)
    application.ui.label_Image.setFont(QFont('Arial', 20))
    application.show()
    QtWidgets.qApp.processEvents()
    application.run()
    application.someFunction()
    sys.exit(app.exec_())
if __name__ == "__main__":
    AnalyzeWindow()
