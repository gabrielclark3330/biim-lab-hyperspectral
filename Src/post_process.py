import pickle
from PIL import Image
from PIL.ImageQt import ImageQt
import os
from pypylon import pylon
import imageio

ipo = pylon.ImagePersistenceOptions()
quality = 100
ipo.SetQuality(quality)
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pick"):
             address = os.path.join(root, file)
             print(address)
             ff = open(address,"rb")
             object_file = pickle.load(ff)
             print(len(object_file))
             for i in range(len(object_file)):
                 filename = "Saved_%s_%d.Jpeg"%(file,i)
                 imageio.imsave(filename,object_file[i])
                 #object_file[i].Save(pylon.ImageFileFormat_Jpeg,filename,ipo)
             ff.close()
