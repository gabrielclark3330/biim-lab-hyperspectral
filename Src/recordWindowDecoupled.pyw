import sys
import cv2
import queue
import os
import shutil
import io
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QDialog, QTabWidget
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QWidget 
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from cv2 import AsyncArray
from pypylon import pylon
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from Arduino import *
import numpy as np
import time as tm
import device
import pickle
import math
import datetime 
import pathlib
import threading
import random 
from PIL import Image
from PIL.ImageQt import ImageQt
from matplotlib.animation import FuncAnimation
import yaml
from yaml import Loader, Dumper
from glob import glob
from scipy.io import savemat
import qdarkstyle
import serial
from RecordWindow_GUI import  Ui_RecordWindow
import subprocess
############################################################
class WatchDogThread(QThread):
    basler_disconnected = pyqtSignal(int)
    #basler_reconnected = pyqtSignal(int)
    watchDogTime=0
    def _init_(self):
        super().__init__()
        self.watchDogTime=0
    def run(self):
        self.watchDogTime=tm.time()
        while(True):
            tm.sleep(1)
            #print("%d\t%d"%(tm.time(),self.watchDogTime))
            if(tm.time()-self.watchDogTime>10):
                self.basler_disconnected.emit(1)
                #global camera_G
                #try:
                #    camera_G=pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                #    self.basler_reconnected.emit(1)
                #    tm.sleep(100)
                #    #camera_G.Close()
                #except:
                #    self.basler_disconnected.emit(1)
                #    print("Camera Is Busy Or Not Connected!")
############################################################
class AnalyzeWindowThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.process=''
    def run(self):
        #print('t4')
        #os.system('analyzeWindowAutoLoad.pyw')
        subprocess.Popen('pythonw.exe analyzeWindowAutoLoad.pyw', shell=True)
        #print('t5')
global grabbingImages
grabbingImages=False
global adjusted
adjusted=False
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        #fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)
        fig.tight_layout()#, h_pad=2.0, w_pad=2.0)
############################################################
class WebCamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.index = 0
        self.camlist = []
        self.autoExposure=True
        self.exposure=10
        self.exposure_old=10
        while True:
            cap = cv2.VideoCapture(self.index)
            if not cap.read()[0]:
                break
            else:
                self.camlist.append(self.index)
            cap.release()
            self.index += 1
        #self.exposure = 0
        self.selectedCam=0
        self.selecetedCamOld = 0
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(self.selectedCam)
        while self._run_flag:
            if(self.selecetedCamOld != self.selectedCam):
                cap = cv2.VideoCapture(self.selectedCam)
            self.selecetedCamOld=self.selectedCam
            if self.autoExposure:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.75)
                self.exposure_old+=1
                #print(cap.get(cv2.CAP_PROP_EXPOSURE))
            else:
                if(not self.exposure==self.exposure_old):
                    self.exposure_old=self.exposure
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE,self.exposure-14)
            ret, cv_img = cap.read()
            
            #cv_img=cv2.flip(cv_img,1)
            if grabbingImages:
                # Save the webcam capture with time as its name into /imageWriteDir
                path = pathlib.Path(__file__).parent.resolve()
                path = os.path.join(path, "imageWriteDir")
                pathExists = os.path.exists(path)
                if not pathExists:
                    os.makedirs(path, exist_ok=True)
                time = datetime.datetime.now().strftime("%H_%M_%S_%f")
                writePath = os.path.join(path, f"{time}.png")
                cv2.imwrite(writePath, cv_img)

            if ret:
                # place cv_img into the user interface video pane
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
global WriteAddress
WriteAddress='.temp'
############################################################
class AsyncWrite(threading.Thread):
    global WriteAddress
    def __init__(self, text , out):
        threading.Thread.__init__(self)
        self.text = text
        self.out = out
    def run(self):
        if(self.out=='reference.pick'):
            f = open(self.out, "wb")
            pickle.dump(self.text,f)
            f.close()
        else:
            f = open(os.path.join(WriteAddress,self.out), "wb")
            pickle.dump(self.text,f)
            f.close()
############################################################
global camera_G
try:
    camera_G=pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
except:
    print("Camera Is Busy Or Not Connected!")
    exit()
class BaslerThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # conecting to the first available camera
    BinningHorizontal = 1        
    BinningVertical = 1
    reverseX=True
    global camera_G
    def __init__(self):
        super().__init__()
        self._run_flag=True
        self.camera = camera_G
        #print("OnOpen event for device ", self.camera.GetDeviceInfo().GetModelName())
        #print("Serial Number", self.camera.GetDeviceInfo().GetSerialNumber())
        self.SN=self.camera.GetDeviceInfo().GetSerialNumber()
        self.exposure = 100
        self.gain = 0.0
        self.gamma = 0.25
        self.record = False
        self.FPS=0.0
        self.dd=[]
        self.ddTimeStamps=[]
        self.Width=0.0
        self.Height=0.0
        self.OffsetX=10
        self.OffsetY=15
        self.setBinning=True
        self.address='./'
        self.frameNum=0
        self.camera.Open()
        self.actualMinExposure=self.camera.AutoExposureTimeLowerLimit.GetMin()
        self.exposureLwLimit=int(self.camera.AutoExposureTimeLowerLimit.GetMin()/1000.0)
        self.exposureUpLimit=min(int(self.camera.AutoExposureTimeUpperLimit.GetMax()/1000.0),1000)
        self.gainLwLimit=self.camera.AutoGainLowerLimit.GetMin()
        self.gainUpLimit=self.camera.AutoGainUpperLimit.GetMax()
        self.camera.Close()
    def run(self):
        # Grabing Continusely (video) with minimal delay
        if self._run_flag:
            self.camera.Open()
            #self.exposureLwLimit=int(self.camera.AutoExposureTimeLowerLimit.GetMin()/1000.0)
            #self.exposureUpLimit=int(self.camera.AutoExposureTimeUpperLimit.GetMax()/1000.0)
            #self.gainLwLimit=self.camera.AutoGainLowerLimit.GetMin()
            #self.gainUpLimit=self.camera.AutoGainUpperLimit.GetMax()
            if(self.setBinning):
                self.setBinning=False
                self.camera.Close()
                self.camera.Open()
                nodemap = self.camera.GetNodeMap()
                #print(self.BinningHorizontal)
                self.camera.BinningHorizontal = self.BinningHorizontal        
                self.camera.BinningVertical = self.BinningVertical
                self.camera.BinningHorizontalMode = "Average"
                self.camera.BinningVerticalMode = "Average" 
                self.camera.Width=self.Width
                self.camera.Height=self.Height
                self.camera.OffsetX=self.OffsetX
                self.camera.OffsetY=self.OffsetY
                #print('Reverse X')
                #print(self.reverseX)
                self.camera.ReverseX.SetValue(self.reverseX)
                #print(self.BinningHorizontal)
                #print(self.BinningVertical)
                #print(self.camera.BinningVerticalMode.GetValue())
                #print(self.camera.BinningHorizontalMode.GetValue())
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            converter = pylon.ImageFormatConverter()
            # converting to opencv bgr format
            converter.InputPixelFormat = pylon.PixelType_Mono16
            converter.OutputPixelFormat = pylon.PixelType_Mono16
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            #print(pylon.ImageFormatConverter_IsSupportedOutputFormat(pylon.PixelType_Mono10))
            self.dd=[]
        while (self.camera.IsGrabbing() and self._run_flag):
        # Grabing Continusely (video) with minimal delay
            WatchDogThread.watchDogTime=tm.time()
            if(self.setBinning):
                self.setBinning=False
                self.camera.StopGrabbing()
                self.camera.Close()
                self.camera.Open()
                nodemap = self.camera.GetNodeMap()
                try:
                    self.camera.BinningHorizontal = self.BinningHorizontal        
                    self.camera.BinningVertical = self.BinningVertical
                    self.camera.BinningVerticalMode = "Average"
                    self.camera.BinningHorizontalMode = "Average"
                    self.camera.Width=self.Width
                    self.camera.Height=self.Height
                    self.camera.OffsetX=self.OffsetX
                    self.camera.OffsetY=self.OffsetY
                except:
                    return
                #print(self.BinningHorizontal)
                #print(self.BinningVertical)
                #print(self.camera.BinningVerticalMode.GetValue())
                #print(self.camera.BinningHorizontalMode.GetValue())
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                converter = pylon.ImageFormatConverter()
                converter.InputPixelFormat = pylon.PixelType_Mono16
                converter.OutputPixelFormat = pylon.PixelType_Mono16
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                # converting to opencv bgr format
                self.dd=[]
            try:
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.ExposureMode.SetValue("Timed")
                self.camera.ExposureTime.SetValue(self.exposure)
                self.camera.GainAuto.SetValue("Off")
                self.camera.Gain.SetValue(self.gain)
                self.camera.Gamma.SetValue(self.gamma)
            except:
                return

            if (self.camera.IsGrabbing() and self._run_flag):
                try:
                    grabResult = self.camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
                    image = converter.Convert(grabResult)
                except:
                    return
            if(self.record):
                i=i+1
                self.frameNum=i
                self.baslerImage = image.GetArray()
                self.dd.append(self.baslerImage.copy())
                time_str = datetime.datetime.now().strftime("%H_%M_%S_%f")
                self.ddTimeStamps.append(str(time_str))
                #pickle.dump(self.baslerImage, open('outfile.pick','a+b'))
                #print(self.baslerImage[0:399,0:1919].shape)
                if (math.fmod(i,100)==0):
                    j=j+1
                    background_write = AsyncWrite(self.dd,'out_%6.6d.pick'%(j))
                    background_write.address=self.address
                    background_write.start()
                    self.dd=[]
                if(math.fmod(i,15)==0):
                    print(str(i),datetime.datetime.now()-dt_old)
                    dt_old=datetime.datetime.now()
            elif(not self.record):
                i=0
                j=0
                self.frameNum=0
                dt_old=datetime.datetime.now()
                    #print(len(self.dd))
            if (self.camera.IsGrabbing() and self._run_flag):
                self.FPS = self.camera.ResultingFrameRate.GetValue()
                if grabResult.GrabSucceeded():
                    # Access the image data
                    image = converter.Convert(grabResult)
                    img = image.GetArray()
                    self.change_pixmap_signal.emit(img)
                    #cv2.namedWindow('title', cv2.WINDOW_NORMAL)
                    #cv2.imshow('title', img)
                    #k = cv2.waitKey(1)
                    #if k == 27:
                    #    break
                grabResult.Release()

    def stop(self):
        # Releasing the resource
        self._run_flag=False
        self.camera.Close()
############################################################
global upFrame
upFrame=0
class LoadingWindow(QtWidgets.QMessageBox):
    def __init__(self):
        super(LoadingWindow, self).__init__()
        self.setWindowTitle("wait")
        self.setText("Record Window Is Loading\nPlease Wait.\n")
    def closeEvent(self, event):
        event.accept()
class RecordApplicationWindow(QtWidgets.QMainWindow):
    resized = QtCore.pyqtSignal()
    def __init__(self):
        super(RecordApplicationWindow, self).__init__()
        QtWidgets.qApp.processEvents()
        msg=LoadingWindow()
        msg.show()
        QtWidgets.qApp.processEvents()
        self.ui = Ui_RecordWindow()
        self.ui.setupUi(self)
        ############################################################
        #configFile=open('config.yaml','r')
        #inputConfig=yaml.load(configFile,Loader=Loader)
        #self.offsetX=int(inputConfig['Offset_X'])
        #self.offsetY=int(inputConfig['Offset_Y'])
        #self.roiX=int(inputConfig['ROI_X'])
        #self.roiY=int(inputConfig['ROI_Y'])
        #self.doubleExposureCoeff=float(inputConfig['Double_Exposure_Coeff'])
        self.readConfig()
        self.readLookUpTable()
        self.maxFrameNum = 200 #TODO: find where this variable is instantiated
        self.ui.spinBox_scanLength.setMaximum(self.maxFrameNum)
        self.ui.spinBox_scanLength.setValue(self.scanLen)
        ############################################################
        self.webcamLabel=self.ui.label_secondaryCamera
        self.webcamThread = WebCamThread()
        for items in self.webcamThread.camlist:
            self.ui.comboBox_secondaryCamera.addItem('%d'%(items))
        self.webcamThread.change_pixmap_signal.connect(self.webcam_update_image)
        self.webcamThread.start()
        self.ui.checkBox_secondaryCameraAutoExposure.setChecked(True)
        self.ui.checkBox_secondaryCameraAutoExposure.stateChanged.connect(self.secondaryAutoExposureChanged)
        self.ui.spinBox_secondaryCameraExposure.valueChanged.connect(self.secondaryExposureChanged)
        self.ui.spinBox_secondaryCameraExposure.setMinimum(0)
        self.ui.spinBox_secondaryCameraExposure.setMaximum(14)
        self.resized.connect(self.someFunction)
        ############################################################
        # create the label that holds the image
        #self.image_label = QLabel(self)
        self.baslerLabel = self.ui.label_BaslerView
        #self.baslerLabel.resize(self.baslerDisplayWidth, self.baslerDisplayHeight)
        self.i=0
        self.showIntensity=True
        self.plot_setted=False
       
       
       
        #self.ui.pushButton_ShowIntensity.setStyleSheet("background-color : green")
        self.tt=False
        self.line=None
        self.set_referance=False
        # create the video capture thread
        BaslerThread.BinningHorizontal = self.binningX
        BaslerThread.BinningVertical =   self.binningY
        #print('tttttttt')
        #print(self.baslerReverseX)
        BaslerThread.reverseX = self.baslerReverseX
        self.baslerThread = BaslerThread()
        self.baslerThread.__init__()
        self.baslerThread.Width=self.roiX
        self.baslerThread.Height=self.roiY
        self.baslerThread.OffsetX=self.offsetX
        self.baslerThread.OffsetY=self.offsetY
        self.ui.label_baslerSN.setText(self.baslerThread.SN)
        # connect its signal to the update_image slot
        self.watchdogThread=WatchDogThread()
        self.watchdogThread.__init__()
        self.watchdogThread.start()
        self.watchdogThread.basler_disconnected.connect(self.baslerDisconnected)
        #self.watchdogThread.basler_reconnected.connect(self.baslerReconnected)
        self.baslerThread.change_pixmap_signal.connect(self.basler_update_image)
        self.lastUpdateBaslerPlotTime=tm.time()
        self.lastUpdateBaslerImageTime=tm.time()
        # start the thread
        self.baslerThread.start()
        self.x=np.arange(0,self.baslerThread.Width,1)
        #self.HighLowExposure=True
        ############################################################
        self.ui.horizontalSlider_Exposure.setMaximum(int(self.baslerThread.exposureUpLimit))
        self.ui.horizontalSlider_Exposure.setMinimum(int(self.baslerThread.exposureLwLimit))
        self.ui.horizontalSlider_Gain.setMaximum(int(self.baslerThread.gainUpLimit)*100)
        self.ui.horizontalSlider_Gain.setMinimum(int(self.baslerThread.gainLwLimit)*100)
        self.ui.horizontalSlider_Gamma.setMaximum(int(self.configGammaMax*100))
        self.ui.horizontalSlider_Gamma.setMinimum(int(self.configGammaMin*100))
        ############################################################
        self.ui.spinBox_exposure.setMaximum(int(self.baslerThread.exposureUpLimit))
        self.ui.spinBox_exposure.setMinimum(int(self.baslerThread.exposureLwLimit))
        self.ui.spinBox_gain.setMaximum(int(self.baslerThread.gainUpLimit)*100)
        self.ui.spinBox_gain.setMinimum(int(self.baslerThread.gainLwLimit)*100)
        self.ui.spinBox_gamma.setMaximum(int(self.configGammaMax*100))
        self.ui.spinBox_gamma.setMinimum(int(self.configGammaMin*100))
            ############################################################
        self.ui.horizontalSlider_Exposure.valueChanged.connect(self.sliderExposureChanged)
        self.ui.horizontalSlider_Gain.valueChanged.connect(self.sliderGainChanged)
        self.ui.horizontalSlider_Gamma.valueChanged.connect(self.sliderGammaChanged)
            ############################################################
        self.ui.spinBox_exposure.valueChanged.connect(self.spinBoxExposureChanged)
        self.ui.spinBox_gain.valueChanged.connect(self.spinBoxGainChanged)
        self.ui.spinBox_gamma.valueChanged.connect(self.spinBoxGammaChanged)
            ############################################################
        self.ui.horizontalSlider_Exposure.setValue(self.setExposure)
        self.ui.horizontalSlider_Gain.setValue(self.setGain)
        self.ui.horizontalSlider_Gamma.setValue(self.setGamma)
        self.ui.spinBox_exposure.setValue(self.setExposure)
        self.ui.spinBox_gain.setValue(self.setGain)
        self.ui.spinBox_gamma.setValue(self.setGamma)
        ############################################################
        self.ui.verticalSlider.setMaximum(self.roiY-1)
        self.ui.verticalSlider.setMinimum(0)
        self.ui.verticalSlider.setSliderPosition(int(self.roiY/2))
        self.ui.pushButton_StartScan.clicked.connect(self.startRecord)
        self.ui.pushButton_StopScan.clicked.connect(self.stopRecord)
        if(os.path.isfile('reference.pick')):
            self.ui.checkBox_referanceStat.setEnabled(True)
            self.ui.checkBox_referanceStat.setChecked(True)
        elif(os.path.isfile('reference.pick.bak')):
            self.ui.checkBox_referanceStat.setEnabled(True)
            self.ui.checkBox_referanceStat.setChecked(False)
        else:
            self.ui.checkBox_referanceStat.setEnabled(False)
        self.ui.checkBox_referanceStat.stateChanged.connect(self.referenceStatChanged)
        self.ui.pushButton_SetRef.clicked.connect(self.set_refrance_plane)
        #self.ui.pushButton_ShowIntensity.clicked.connect(self.showIntensityMethod)
        self.ui.pushButton_SaveSetting.clicked.connect(self.writeConfig)
    ############################################################
        if(not self.plot_setted):
            self.canvas = MplCanvas(self, width=5, height=2, dpi=100)
            self.canvas.axes.set_ylim(0,70000/2**4)
            self.canvas.axes.set_xlim(400,950)
            self.canvas.axes.set_xlabel('(nm)',color='white',labelpad=-10)
            self.canvas.axes.yaxis.grid(True,linestyle='--',color='red')
            self.canvas.axes.xaxis.grid(True,linestyle='--',color='red')
            self.major_ticks_x = np.arange(400,1000, 50)
            self.canvas.axes.set_xticks(self.major_ticks_x)
            #self.minor_ticks_x = np.arange(400,1000, 10)
            #self.major_ticks_y = np.arange(0,70000/2**4, 1000)
            #self.minor_ticks_y = np.arange(0,70000/2**4, 200)
            self.canvas.axes.grid(which='minor', alpha=0.2)
            #self.canvas.axes.grid(which='major', alpha=0.5)
            #self.canvas.axes.set_xticks(self.minor_ticks_x, minor=True)
            #self.canvas.axes.set_yticks(self.major_ticks_y)
            #self.canvas.axes.set_yticks(self.minor_ticks_y, minor=True)
            self.canvas.axes.tick_params(axis='both',labelcolor="white",color="white")
            self.canvas.axes.spines['top'].set_color("white")
            self.canvas.axes.spines['left'].set_color("white")
            self.canvas.axes.spines['right'].set_color("white")
            self.canvas.axes.spines['bottom'].set_color("white")
            self.canvas.axes.set_facecolor((25.0/255.0,35.0/255.0,45.0/255.0))
            self.canvas.figure.set_facecolor((25.0/255.0,35.0/255.0,45.0/255.0))
            global adjusted
            if(not adjusted):
                self.canvas.figure.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
                adjusted = True
        #self.canvas.figure.tight_layout(pad=0.2, h_pad=-.5, w_pad=-5)
        #self.canvas.axes.tick_params(axis="y",direction="in", pad=-30)
        self.ui.gridLayout.addWidget(self.canvas, 11, 2, 3, 10)
        #self.IntensityWidget.add(self.canvas,2,1,1,1)
    ############################################################
        self.ui.comboBox_ArduinoCom.currentTextChanged.connect(self.setArduinoCom)
    ############################################################
        self.timeSetted=False
        self.timeRecordStart=tm.time()
#        self.ui.comboBox_biningX.currentTextChanged.connect(self.on_combobox_changed)
#        self.ui.comboBox_biningY.currentTextChanged.connect(self.on_combobox_changed)
#    def on_combobox_changed(self, value):
#        BaslerThread.BinningHorizontal = int(self.ui.comboBox_biningX.currentText()) 
#        BaslerThread.BinningVertical =   int(self.ui.comboBox_biningY.currentText()) 
#        self.baslerThread.setBinning=True
        self.disConnectedMsgStat=False
    def baslerDisconnected(self,stat):
        if(bool(stat)):
            if(not self.disConnectedMsgStat):
                self.disConnectedMsgStat=True
                qm = QMessageBox()
                qm.setIcon(QMessageBox.Warning)
                qm.about(self,'Error','HySpim is Disconnected!\n Exiting!')
                exit()
    #def baslerReconnected(self,stat):
    #    self.watchdogThread.watchDogTime=tm.time()
    #    self.disConnectedMsgStat=False
    #    print('RE')
    #    if(bool(stat)):
    #        print('RE2')
    #        self.baslerThread._run_flag=False
    #        self.baslerThread.stop()
    #        BaslerThread.BinningHorizontal = self.binningX
    #        BaslerThread.BinningVertical =   self.binningY
    #        #print('tttttttt')
    #        #print(self.baslerReverseX)
    #        BaslerThread.reverseX = self.baslerReverseX
    #        self.baslerThread = BaslerThread()
    #        self.baslerThread.__init__()
    #        self.baslerThread.Width=self.roiX
    #        self.baslerThread.Height=self.roiY
    #        self.baslerThread.OffsetX=self.offsetX
    #        self.baslerThread.OffsetY=self.offsetY
    #        self.baslerThread.start()

    def referenceStatChanged(self, event):
        if self.ui.checkBox_referanceStat.isChecked():
            if(os.path.isfile('reference.pick.bak')):
                os.rename('reference.pick.bak','reference.pick')
        elif not self.ui.checkBox_referanceStat.isChecked():
            if(os.path.isfile('reference.pick')):
                os.rename('reference.pick','reference.pick.bak')
    def secondaryExposureChanged(self,event):
        self.webcamThread.exposure=self.ui.spinBox_secondaryCameraExposure.value()
    def secondaryAutoExposureChanged(self,event):
        if self.ui.checkBox_secondaryCameraAutoExposure.isChecked():
            self.webcamThread.autoExposure=True
        else:
            self.webcamThread.autoExposure=False
            self.webcamThread.exposure=self.ui.spinBox_secondaryCameraExposure.value()
    def sliderExposureChanged(self,event):
        self.ui.spinBox_exposure.setValue(self.ui.horizontalSlider_Exposure.value())
    def spinBoxExposureChanged(self,event):
        self.ui.horizontalSlider_Exposure.setValue(self.ui.spinBox_exposure.value())
    def sliderGainChanged(self,event):
        self.ui.spinBox_gain.setValue(self.ui.horizontalSlider_Gain.value())
    def spinBoxGainChanged(self,event):
        self.ui.horizontalSlider_Gain.setValue(self.ui.spinBox_gain.value())
    def sliderGammaChanged(self,event):
        self.ui.spinBox_gamma.setValue(self.ui.horizontalSlider_Gamma.value())
    def spinBoxGammaChanged(self,event):
        self.ui.horizontalSlider_Gamma.setValue(self.ui.spinBox_gamma.value())

    def resizeEvent(self, event):
        self.resized.emit()
        return super(RecordApplicationWindow, self).resizeEvent(event)
    def someFunction(self):
        self.webCamDisplayWidth =self.ui.label_secondaryCamera.width()
        self.webCamDisplayHeight = self.ui.label_secondaryCamera.height()
        self.baslerDisplayWidth = self.ui.label_BaslerView.width()
        self.baslerDisplayHeight = self.ui.label_BaslerView.height()
    #############################################################
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
        f=open('config.bin','wb')
        for num in nums:
            #print(format(num,'020b'))
            f.write(format(num,'0100b').encode('ASCII'))
        f.close()
    def readConfig(self):
        f=open('config.bin','r')
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
    ###########################################################
    def readLookUpTable(self):
        lookupFile=open('lookupTable.txt','r')
        self.x_lookUp=[]
        self.y_lookUp=[]
        for lines in lookupFile:
            self.x_lookUp.append(float(lines.split()[0]))
            self.y_lookUp.append(float(lines.split()[1]))
    ############################################################
    def setArduinoCom(self,com):
        #print('Com is :'+com)
        if(com != ''):
            try:
                self.ser=serial.Serial(com,9600,timeout=10)
            except:
                self.ser.close()
                self.ser=serial.Serial(com,9600,timeout=10)
    ############################################################
    @pyqtSlot(np.ndarray)
    def webcam_update_image(self, webcam_cv_img):
        """Updates the image_label with a new opencv image"""
        #self.webcamThread.exposure=self.ui.exposureSlider.value()/100.0
        self.webcamThread.selectedCam=int(self.ui.comboBox_secondaryCamera.currentText())
        if(self.flipSecondaryVertical):
            webcam_cv_img=cv2.flip(webcam_cv_img,0)
        if(self.flipSecondaryHorizontal):
            webcam_cv_img=cv2.flip(webcam_cv_img,1)
        webcam_cv_img[0::2,self.secondaryVerticalLine:self.secondaryVerticalLine+3]=1000000
        webcam_cv_img[1::2,self.secondaryVerticalLine:self.secondaryVerticalLine+3]=-1000000
        qt_img = self.convert_cv_qt_webCam(webcam_cv_img)
        self.webcamLabel.setPixmap(qt_img)
    def convert_cv_qt_webCam(self, webcam_cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(webcam_cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.webCamDisplayWidth, self.webCamDisplayHeight, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    ################################################################
    def closeEvent(self, event):
        print('closing')
        self.baslerThread._run_flag=False
        self.baslerThread.stop()
        self.webcamThread.stop()
        event.accept()
    def set_refrance_plane(self):
        self.ui.pushButton_SetRef.setStyleSheet("background-color : red")
        self.set_referance=True
        self.startRecord()
    def showIntensityMethod(self):
        self.showIntensity = not self.showIntensity
        #if(self.showIntensity):
        #    self.ui.pushButton_ShowIntensity.setStyleSheet("background-color : green")
        #else:
        #    self.ui.pushButton_ShowIntensity.setStyleSheet("background-color : red")
    @pyqtSlot(np.ndarray)
    def basler_update_image(self, basler_cv_img):
        self.watchdogThread.watchDogTime=tm.time()
        if(self.baslerThread.frameNum==self.refFrameNum and self.set_referance):
            self.stopRecord()
            self.ui.pushButton_SetRef.setStyleSheet("background-color : green")
            self.ui.checkBox_referanceStat.setChecked(True)
            self.ui.checkBox_referanceStat.setEnabled(True)
        if(self.baslerThread.frameNum==self.ui.spinBox_scanLength.value() and (not self.ui.spinBox_scanLength.value()==0)):
            self.stopRecord()
            #analyzeThread=AnalyzeWindowThread()
            #analyzeThread.__init__()
            #analyzeThread.start()
        if(self.ui.spinBox_scanLength.value()==0 and self.baslerThread.frameNum==1):
            self.stopRecord()
            #analyzeThread=AnalyzeWindowThread()
            #analyzeThread.__init__()
            #analyzeThread.start()
        """Updates the image_label with a new opencv image"""
        ################################################################
        self.ui.label_FPS.setText('FPS: %f'%self.baslerThread.FPS)
        #self.ui.label_Exposure.setText('%d'%self.ui.horizontalSlider_Exposure.value())
        #self.ui.label_Gain.setText('%f'%(self.ui.horizontalSlider_Gain.value()/10.0))
        #self.ui.label_Gamma.setText('%f'%(self.ui.horizontalSlider_Gamma.value()/50.0))
        if(self.timeSetted):
            self.ui.label_recordTime.setText(f'Record Time: {round(tm.time()-self.timeRecordStart, 1)} (Sec) / {round(self.ui.spinBox_scanLength.value()/self.baslerThread.FPS, 1)} (sec)')
        else:
            self.ui.label_recordTime.setText(f'Record Time: {0.0} (Sec) / {round(self.ui.spinBox_scanLength.value()/self.baslerThread.FPS, 1)} (sec)')
        ################################################################
        self.baslerThread.gain=self.ui.horizontalSlider_Gain.value()/100.0
        self.baslerThread.gamma=self.ui.horizontalSlider_Gamma.value()/100.0
        #if(self.HighLowExposure):
        self.baslerThread.exposure=max(self.ui.horizontalSlider_Exposure.value()*1000,self.baslerThread.actualMinExposure)
        #self.baslerThread.exposure=min(self.baslerThread.exposure,1000*1000)
         #   self.HighLowExposure=False
        #else:
        #    self.baslerThread.exposure=int(self.ui.horizontalSlider_Exposure.value()*self.doubleExposureCoeff)
        #    self.HighLowExposure=True
        #    return

        ################################################################
        sliderValue=self.roiY-1-self.ui.verticalSlider.value()
        if(self.showIntensity):
            if ((tm.time()-self.lastUpdateBaslerPlotTime)>0.1E0):
                self.lastUpdateBaslerPlotTime=tm.time()
                if(not self.plot_setted):
                    self.plot_setted=True
                    self.canvas.axes.tick_params(axis='both',labelcolor="white",color="white")
                    self.canvas.axes.yaxis.grid(True,linestyle='--',color='red')
                    self.canvas.axes.xaxis.grid(True,linestyle='--',color='red')
                    self.canvas.axes.spines['top'].set_color("white")
                    self.canvas.axes.spines['left'].set_color("white")
                    self.canvas.axes.spines['right'].set_color("white")
                    self.canvas.axes.spines['bottom'].set_color("white")
                    self.canvas.axes.set_xticks(self.major_ticks_x)
                    #self.canvas.axes.set_xticks(self.minor_ticks_x, minor=True)
                    #self.canvas.axes.set_yticks(self.major_ticks_y)
                    #self.canvas.axes.set_yticks(self.minor_ticks_y, minor=True)
                    for i in range(20,1,-1):
                        if(len(basler_cv_img[sliderValue,:])%i==0):
                            self.CG=i
                            break
                    #print('CG is %d'%self.CG)
                n=self.CG
                list2 = [sum(basler_cv_img[sliderValue,i:i+n])//(n*2**4) for i in range(0,len(basler_cv_img[sliderValue,:]),n)]
                xx=np.linspace(self.y_lookUp[0],self.y_lookUp[-1],num=len(list2),endpoint=False,dtype=float)
                self.canvas.axes.set_facecolor((25.0/255.0,35.0/255.0,45.0/255.0))
                self.canvas.axes.set_ylim(0,70000/2**4)
                self.canvas.axes.set_xlim(400,950)
                self.canvas.axes.set_xlabel('(nm)',color='white',labelpad=-10)
                if self.line is None:
                    plot_ref = self.canvas.axes.plot(xx,list2,color='yellow')
                    self.line = plot_ref[0]
                else:
                    self.line.set_ydata(list2)
                self.canvas.axes.yaxis.grid(True,linestyle='--')
                self.canvas.axes.xaxis.grid(True,linestyle='--')
                self.canvas.draw()
        else:
            self.canvas.axes.set_ylim(0,70000/2**4)
            self.canvas.axes.set_xlim(400,950)
            self.canvas.axes.set_xlabel('(nm)',color='white',labelpad=-10)
            self.canvas.axes.clear()
            #self.canvas.axes.cla()
            self.line=None
        #if(not self.HighLowExposure):
        if((tm.time()-self.lastUpdateBaslerImageTime)>0.33):
            basler_cv_img[sliderValue-4:sliderValue+4,1::2]=0
            basler_cv_img[sliderValue-4:sliderValue+4,0::2]=2**16-1
            qt_img = self.convert_cv_qt_basler(basler_cv_img)
        #qt_img = qt_img.copy(0,0, self.baslerDisplayWidth, self.baslerDisplayHeight)
            self.lastUpdateBaslerImageTime=tm.time()
            self.baslerLabel.setPixmap(qt_img)
            self.baslerLabel.setAlignment(Qt.AlignCenter)
    def convert_cv_qt_basler(self, basler_cv_img):
        """Convert from an opencv image to QPixmap"""
        #rgb_image = cv2.cvtColor(basler_cv_img, cv2.COLOR_BGR2RGB)
        #h, w, ch = rgb_image.shape
        #print(h,w,ch)
        #bytes_per_line = ch * w
        h,w=np.shape(basler_cv_img)
        #print(h,w)
        convert_to_Qt_format = QtGui.QImage((basler_cv_img/256).astype('uint8'),w,h,w,QtGui.QImage.Format_Grayscale8)
        #convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format.scaled(self.baslerDisplayWidth-int(self.baslerDisplayWidth*0.1),self.baslerDisplayHeight))#,Qt.KeepAspectRatio) )
    def startRecord(self):
        ''' @TAGCOM
        try:
            self.ser.write(b'\r\n')
        except serial.serialutil.SerialException:
            self.ui.comboBox_ArduinoCom.clear()
            qm = QMessageBox()
            qm.setIcon(QMessageBox.Warning)
            qm.about(self,'Error','Scanner Not Found!')
            try:
                self.ser=serial.Serial(com,9600,timeout=10)
            except:
                qm = QMessageBox()
                qm.setIcon(QMessageBox.Warning)
                qm.about(self,'Error','Please Connect Scanner\nAnd\nUpdate Scanner Port!')
            return
        '''
        if(not self.set_referance):
            for i in range(4):
                #self.ser.write(b'\r\n') 
                time.sleep(0.1)
            #self.ser.write(b'203\r\n')
            #print('Back To Home')
            time.sleep(0.5)
            global WriteAddress
            if(os.path.exists(WriteAddress) and os.path.isdir(WriteAddress)):
                files=glob(WriteAddress)
                for f in files:
                    shutil.rmtree(f)
                os.mkdir(WriteAddress)
            else:
                os.mkdir(WriteAddress)
            if(not os.path.isfile('reference.pick')):
                qm = QMessageBox()
                qm.setIcon(QMessageBox.Warning)
                ret = qm.question(self,'', "No Reference Image Found\nContinue?", qm.Yes | qm.No)
                if (ret == qm.Yes):
                    tempVar=1
                elif(ret==qm.No):
                    self.showIntensity = True
                    self.plot_setted=False
                    return
            self.move_it()
        else:
            if(os.path.isfile('reference.pick')):
                os.remove('reference.pick')
            if(os.path.isfile('reference.pick.bak')):
                os.remove('reference.pick.bak')
            self.baslerThread.record = True
            global grabbingImages
            grabbingImages = True
        self.showIntensity = False
            #self.ui.pushButton_ShowIntensity.setStyleSheet("background-color : red")
        #LocalWriteAddress=self.getAddress()
        #if LocalWriteAddress is None:
        #    return
        #else:
        #    WriteAddress = LocalWriteAddress 
        if (not self.timeSetted):
            self.timeSetted=True
            self.timeRecordStart=tm.time()
    def move_it(self):
        #self.ser.flush()
        fp=self.baslerThread.FPS
        x=int(10.0*self.maxFrameNum/fp)
        x=(1.E+6/(fp*400))+self.maxFrameNum
        #self.ser.write(b'%d\r\n'%x)
        #self.ser.flush()
        time.sleep(0.5)
        x=201
        #self.ser.write(b'%d\r\n'%x)
        time.sleep(0.5)
        x=self.ui.spinBox_scanLength.value()
        #self.ser.write(b'%d\r\n'%x)
        time.sleep(0.5)
        self.baslerThread.record = True
        global grabbingImages
        grabbingImages = True
        self.showIntensity = False
        #self.ui.pushButton_ShowIntensity.setStyleSheet("background-color : red")
    def getAddress(self):
        LocalWriteAddress = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if LocalWriteAddress=='':
            return None
        if (not len(os.listdir(LocalWriteAddress)) == 0):
            qm = QMessageBox
            ret=qm.question(self,'', "Selected Directory :\n"+LocalWriteAddress+"\nIs Not Empty\nData Corruption Is Possible\nContinue?", qm.Yes | qm.No)
            if ret==qm.Yes:
                return LocalWriteAddress 
            else:
                self.getAddress()
        else:
            return LocalWriteAddress 
    def stopRecord(self):
        global WriteAddress
        self.showIntensity = True
        self.plot_setted=False
        self.baslerThread.record = False
        global grabbingImages
        grabbingImages = False
        self.timeSetted=False
        if(self.set_referance):
            background_write = AsyncWrite(self.baslerThread.dd,'reference.pick')
            background_write.start()
            self.set_referance=False
            self.baslerThread.dd=[]
        else:
            background_write = AsyncWrite(self.baslerThread.dd,'out_final.pick')
            background_write.start()

            filename = "./.tempTime/out_final_timestamps.pick"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            f = open(filename, "wb")
            pickle.dump(self.baslerThread.ddTimeStamps,f)
            f.close()

            self.baslerThread.ddTimeStamps=[]
            self.baslerThread.dd=[]
            analyzeThread=AnalyzeWindowThread()
            analyzeThread.__init__()
            self.watchdogThread.watchDogTime=tm.time()
            analyzeThread.start()

        #self.ser.write(b'203\r\n') @TAGCOM
        #self.ser.write(b'203\r\n')
        #self.ser.write(b'203\r\n')
        #self.ser.write(b'203\r\n')
        #self.ser.write(b'203\r\n')
        #self.ser.write(b'203\r\n')
def updateSerialPorts(application):
    application.ui.comboBox_ArduinoCom.clear()
    portList=serial_ports()
    for items in portList:
        application.ui.comboBox_ArduinoCom.addItem(items)
    ################################################################
def recordWindow():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    application = RecordApplicationWindow()
    application.resize(1024,800)
    portList=serial_ports()
    cb = application.ui.comboBox_ArduinoCom
    for items in portList:
        cb.addItem(items)
    application.ui.pushButton_Update.clicked.connect(lambda: updateSerialPorts(application))
    application.show()
    dd=[]
    application.someFunction()
    sys.exit(app.exec_())
if __name__ == "__main__":
    recordWindow()
