import time 
from pypylon import pylon
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

# When recording this code assumes the camera is pointing down, the silver pin forward, the HSI logo to the right, and the camera being moved to the right
# the returned matrix contains line scans in each column and each row as a different frequency
camera.Open()
camera.ExposureAuto.SetValue("Off")
camera.ExposureMode.SetValue("Timed")
camera.ExposureTime.SetValue(50000) # <- good for images when the lights on included platform are on
camera.GainAuto.SetValue("Off")
camera.Gain.SetValue(5)
camera.Gamma.SetValue(2)
converter = pylon.ImageFormatConverter()
converter.InputPixelFormat = pylon.PixelType_Mono16
converter.OutputPixelFormat = pylon.PixelType_Mono16
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

fps = camera.ResultingFrameRate.GetValue()
print("fps: ", fps)

hyper_cube = []
time_stamps = []

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
    image = converter.Convert(grabResult)
    basler_image = image.GetArray()
    hyper_cube.append(basler_image[:,250:850])
    time_stamps.append((time.time(), time.time_ns()))
    cv2.imshow('hypim image', basler_image)
    if cv2.waitKey(1) == ord('q'):
        break


plt.imshow(hyper_cube[50])
plt.show()


hyper_cube = np.array(hyper_cube)
display = hyper_cube[:,:,176]
plt.imshow(display)
plt.show()


with open("hyperspectralData.pick", "wb") as f:
    pickle.dump(np.array(hyper_cube).tolist(), f)
    #json.dump(np.array(hyper_cube).tolist(), f) #
with open("hyperspectralTimestamps.pick", "wb") as f:
    pickle.dump(time_stamps, f)
    #json.dump(time_stamps, f)

