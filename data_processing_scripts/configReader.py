import numbers
import numpy as np
import os


path = os.path.join(os.getcwd(), 'Src\\config.bin')
f=open(path,'r')
nums=f.read()
calibCoeff=np.zeros(shape=(6),dtype=float)
offsetX=int(nums[0:100],2)
offsetY=int(nums[100:200],2)
roiX=int(nums[200:300],2)
roiY=int(nums[300:400],2)
binningX=int(nums[400:500],2)
binningY=int(nums[500:600],2)
HighRes=bool(int(nums[600:700],2))
doubleExposureCoeff=float(int(nums[700:800],2))/1000.0
calibCoeff[0]=float(int(nums[800:900],2))/1000.0
calibCoeff[1]=float(int(nums[900:1000],2))/1000.0
calibCoeff[2]=float(int(nums[1000:1100],2))/1000.0
calibCoeff[3]=float(int(nums[1100:1200],2))/1000.0
calibCoeff[4]=float(int(nums[1200:1300],2))/1000.0
calibCoeff[5]=float(int(nums[1300:1400],2))/1000.0
offAxisPixNum=int(nums[1400:1500],2)
CWRotation=bool(int(nums[1500:1600],2))
refFrameNum=int(nums[1600:1700],2)
setExposure=int(nums[1700:1800],2)
setGain=int(nums[1800:1900],2)
setGamma=int(nums[1900:2000],2)
scanLen=int(nums[2000:2100],2)
maxFrameNum=int(nums[2100:2200],2)
flipSecondaryVertical=bool(int(nums[2200:2300],2))
flipSecondaryHorizontal=bool(int(nums[2300:2400],2))
secondaryVerticalLine=int(nums[2400:2500],2)
order=int(nums[2500:2600],2)
baslerReverseX=bool(int(nums[2600:2700],2))
configGammaMin=float(int(nums[2700:2800],2))/1000.0
configGammaMax=float(int(nums[2800:2900],2))/1000.0
f.close()

print(maxFrameNum)
### CHANGES ###
#maxFrameNum = 200
###############

doubleExposureCoeff=int(doubleExposureCoeff*1000)
calibCoeff[0]=int(calibCoeff[0]*1000)
calibCoeff[1]=int(calibCoeff[1]*1000)
calibCoeff[2]=int(calibCoeff[2]*1000)
calibCoeff[3]=int(calibCoeff[3]*1000)
calibCoeff[4]=int(calibCoeff[4]*1000)
calibCoeff[5]=int(calibCoeff[5]*1000)
configGammaMin=int(configGammaMin*1000)
configGammaMax=int(configGammaMax*1000)

nums=[
    int(offsetX),
    int(offsetY),
    int(roiX),int(roiY),
    int(binningX),
    int(binningY),
    int(HighRes),
    doubleExposureCoeff,
    calibCoeff[0],
    calibCoeff[1],
    calibCoeff[2],
    calibCoeff[3],
    calibCoeff[4],
    calibCoeff[5],
    int(offAxisPixNum),
    int(CWRotation),
    int(refFrameNum),
    int(setExposure),
    int(setGain),
    int(setGamma),
    int(scanLen),
    int(maxFrameNum),
    int(flipSecondaryVertical),
    int(flipSecondaryHorizontal),
    int(secondaryVerticalLine), 
    int(order),
    int(baslerReverseX),
    configGammaMin,
    configGammaMax
]

def numberToBase(n, b):
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    binaryString = ""
    for digit in digits:
        binaryString = str(digit) + binaryString
    return binaryString

# NOTE: every number should have 100 binary digits
def formatBinaryNumberString(numberString):
    formattedString = ""

    for i in range(0, 100-len(numberString)):
        formattedString = formattedString + "0"

    formattedString = formattedString + numberString
    return formattedString




path = os.path.join(os.getcwd(), 'Config\\config.bin')
f=open(path,'wb')
for num in nums:
    print(num)
    #print(numberToBase(num, 2))
    #f.write(formatBinaryNumberString(numberToBase(num,2)).encode('ASCII'))
f.close()
