# import the necessary packages
import os
import pathlib
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
path = pathlib.Path(__file__).parent.resolve()
path = os.path.join(path, "imageWriteDir")
print(path)
imagePaths = sorted(list(paths.list_images(path)))
images = []
# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)

# initialize OpenCV's image stitcher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
	# write the output stitched image to disk
    pathO = pathlib.Path(__file__).parent.resolve()
    pathO = os.path.join(path, "stitchedImage.png")
    cv2.imwrite(pathO, stitched)
	# display the output stitched image to our screen
    # cv2.imshow("Stitched", stitched)
    # cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))