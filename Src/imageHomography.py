# import the necessary packages
import os
import pathlib
from re import I
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
from matplotlib import pyplot as plt

# Vector and homography goes from img1->img2
# TimeStamp is for img1
class transform():
    def __init__(self, vector, homography, timeStamp):
        self.vector = vector
        self.homography = homography
        self.timeStamp = timeStamp

def findVectorFromTo(from_set, to_set):
    avg_point_from = [0 for x in range(len(from_set[0][0]))]
    avg_point_to = [0 for x in range(len(to_set[0][0]))]
    for point in from_set:
        point = point[0]
        for index, component in enumerate(point):
            avg_point_from[index] += component/len(from_set)
    for point in to_set:
        point = point[0]
        for index, component in enumerate(point):
            avg_point_to[index] += component/len(to_set)

    vector = [0 for x in range(len(avg_point_to))]
    for component in range(len(avg_point_to)):
        vector[component] = avg_point_to[component] - avg_point_from[component]

    return vector

# Stores data cube into a 3dimentional array with images being stored on the yz axis and each x layer representing a wavelength
# z querys the linescans
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
    dimA=np.shape(np_data_cube)[0]          # 341 or #of wavelength buckets
    dimB=int(np.shape(np_data_cube)[1]/200) # height of scan? <- has to be I think
    dimC=200                                # len of scan

    cube_file=np.zeros(shape=(dimA,dimB,dimC),dtype=int)
    cube_file=np.reshape(np_data_cube,(dimA,dimB,dimC))
    return cube_file

### GET ALL IMAGES FROM PATH AND APPEND TO IMAGES ###
path = pathlib.Path(__file__).parent.resolve()
path = os.path.join(path, "imageWriteDir")
print("[INFO] loading images from: ", path)
imagePaths = sorted(list(paths.list_images(path)))
images = []
# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
    timestamp = os.path.basename(os.path.normpath(imagePath))[0:-4] # strips the .png ending off the string
    image = cv2.imread(imagePath, 0) # grayscale specified
    images.append((image, timestamp))
#####################################################
print("[INFO] images loaded")

def getTransfromImgs(img1, img2):
    timestamp = img1[1]
    img1 = img1[0]
    img2 = img2[0]
    ### CALCULATE MATCHING KEY POINTS BETWEEN IMAGES ###
    MIN_MATCH_COUNT = 9
    #img1 = images[0] # queryImage
    #img2 = images[40] # trainImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    #####################################################

    ### CALCULATE HOMOGRAPHY BETWEEN IMAGES ###
    homography = None
    v_p1_to_p2 = None
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        v_p1_to_p2 = findVectorFromTo(src_pts, dst_pts)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,homography)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        return None
    #####################################################
    ### DISPLAY MATCHING IMAGE POINTS ###
    ''' 
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()
    '''
    return transform(v_p1_to_p2, homography, timestamp)
    #####################################################


'''
Algorithm steps:
1. Find the closest web cam images to each hyperspectral line scan and pair them
    Do a binary search through the image timestamps for each linescan
2. Get the transform(vector, homography) from each webcam image to the next image in order of time.
3. For each line scan, map each of its points from 
    ((lineScan).y+len(image)/2)
    ((lineScan).x+len(image[0])/2)
    lineScan*homography # all points in line scan are shifted by homorgraphy
    lineScan # all points in line scan are shifted by all vectors up to that timestamp added together
    store point data with its corisponding hyperspectral data
4. Print stored data into image
'''

def findClosestTransform(time, trasnforms):
    closest_transform = None
    return closest_transform


cube_file = readDataCube('.\\data\\stitchingTestDesk.cube') # [spectrum, linescan[i], linescan]
ff = open('.\\Src\\.tempTime\\timestamps_17_00_19.pick',"rb")
line_scan_times = pickle.load(ff)

# transformArr is filtered based on if there were enough matches between two images
transformArr = []
img1 = images[0]
for img2 in images[1:]:
    transf = getTransfromImgs(img1, img2)
    if transf:
        transformArr.append(transf)
        img1 = img2



x = []
y = []
currentPos = [0 for x in range(len(transformArr[0].vector))]
for transf in transformArr:
    for component in range(len(transf.vector)):
        currentPos[component] += transf.vector[component]
    x.append(currentPos[0])
    y.append(currentPos[1])
print(transformArr)
plt.scatter(x, y),plt.show()



''' 
### CALCULATE HOMOGRAPHY BETWEEN IMAGES ###
# Read source image.
im_src = images[0]
pts_src = np.array()
# Read destination image.
im_dst = images[1]
pts_dst = np.array()
# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)
# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
# Display images
cv2.imshow("Source Image", im_src)
cv2.imshow("Destination Image", im_dst)
cv2.imshow("Warped Source Image", im_out)
cv2.waitKey(0)
#####################################################
'''

