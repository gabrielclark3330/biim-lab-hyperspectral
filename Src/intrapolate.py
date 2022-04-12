import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from time import time
 
img = cv2.imread('Saved_out_final.pick_0.Jpeg')
print(img.shape)
#print(img[:,:,0])
#print(img[:,:,1])
#print(img[:,:,2])qq
#img[:,:,1]=0
#img[:,:,0]=0
#img[:,:,2]=0

#M=447
#N=995
#m=240

#ImWidth=979
#ImHeight=458
ROI_X=0
ROI_Y=0
ROI_Width=1920
ROI_Height=400
outOfPlane=20
ROI_Img=img[ROI_Y:ROI_Y+ROI_Height,ROI_X:ROI_X+ROI_Width,:]
ImHeight, ImWidth,RGBDim = ROI_Img.shape


#j=-1
#for i in img[:,0,0]:
#    j += 1
#    print(j,img[j,0,0])
cv2.imshow('main',ROI_Img)
cv2.waitKey(0) 
tt1=time()
bicubic_img = cv2.resize(ROI_Img,None, fx = 1, fy = (ImWidth+0.0)/outOfPlane, interpolation = cv2.INTER_CUBIC)
tt2=time()
print(tt2-tt1)
#cv2.imshow('main',bicubic_img)
#cv2.waitKey(0) 

mv_img=bicubic_img.copy()
mv_img[:,:,:]=0

IntImHeight, IntImWidth , RGBDim = bicubic_img.shape
for i in range(IntImWidth):
    #print(N-i)
    mv_img[0:IntImHeight-i,i,:]=bicubic_img[i:IntImHeight,i,:]
    #print(i,N-i,bicubic_img[i:1857,N-i,0])
    #print(i,N-i,bicubic_img[i:1857,N-i,1])
    #print(i,N-i,bicubic_img[i:1857,N-i,2])

mv_img_2=mv_img[0:(IntImHeight-IntImWidth),:,:]
final_img = cv2.resize(mv_img_2,None, fx = 1, fy = outOfPlane/ImWidth, interpolation = cv2.INTER_CUBIC)
tt3=time()
print(tt3-tt1)

cv2.imshow('main2',final_img)
cv2.waitKey(0)
cv2.imwrite('im2.jpg',final_img)
