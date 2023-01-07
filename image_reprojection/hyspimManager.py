import time 
from pypylon import pylon
import cv2

time_sec = time.time()
time_nsec = time.time_ns()

camera = None
try:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    '''
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    for device in devices:
        print(device.GetFriendlyName())
    '''
except Exception as e:
    print("Camera Is Busy Or Not Connected!" + str(e))
    exit()

camera.Open()
camera.ExposureAuto.SetValue("Off")
camera.ExposureMode.SetValue("Timed")
camera.ExposureTime.SetValue(50000)
camera.GainAuto.SetValue("Off")
camera.Gain.SetValue(25)
camera.Gamma.SetValue(1)
converter = pylon.ImageFormatConverter()
converter.InputPixelFormat = pylon.PixelType_Mono16
converter.OutputPixelFormat = pylon.PixelType_Mono16
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

fps = camera.ResultingFrameRate.GetValue()
print("fps: ", fps)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    image = converter.Convert(grabResult)
    basler_image = image.GetArray()
    cv2.imshow('hypim image', basler_image)
    if cv2.waitKey(1) == ord('q'):
        break
    #print(basler_image)