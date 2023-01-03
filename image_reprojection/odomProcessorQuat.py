import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import cv2
import quaternion

'''
NOTES:

from the cameras perspective,
fowards: -y
backwards: y
right: -x
left: x
up: z
down: -z
'''

### START HELPER FUNCTIONS ###
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction; (0,1,0 the position vector for c2 and the position vector for e in the global coordinate system then all yo)
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    p_co = np.array(p_co)
    p_no = np.array(p_no)

    u = p1+p0
    dot = np.dot(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = p0-p_co
        fac = -np.dot(p_no, w) / dot
        u = u*fac
        return p0+u

    # The segment is parallel to plane.
    return None

# remaps a value in range r1 into a returned value in range r2
def remap_range(value, r1, r2): # ranges are inclusive on both sides
    return r2[0] + (value - r1[0]) * (r2[1] - r2[0]) / (r1[1] - r1[0])

class OdomDatum():
    # position in [x,y,z] orientation in [x,y,z,w]
    def __init__(self, sec, nsec, position, orientation):
        self.sec = sec
        self.nsec = nsec
        self.position = position
        self.orientation = orientation 

class HYSPMDatum():
    # linescan is an MxN matrix where M is and N is # TODO: fill in description
    def __init__(self, sec, nsec, linescan, odom_match_index=None):
        self.sec = sec
        self.nsec = nsec
        self.linescan = linescan
        self.odom_match_index = odom_match_index
### END HELPER FUNCTIONS ###


### START READ IN ODOM FILE ###
odom_file = open('./odom.txt', 'r')
lines = odom_file.readlines()
file_lines = []
for line in lines:
    file_lines.append(line)
### END READ IN ODOM FILE ###


### START PARSE ODOM FILE INTO ARRAY AND OBJECT ###
odometry_data = []
for index, line in enumerate(file_lines):
    if line[:7]=="header:" and index+17 < len(file_lines):
        sec = int(file_lines[index+3][file_lines[index+3].find(":")+2:-1])
        nsec = int(file_lines[index+4][file_lines[index+4].find(":")+2:-1])
        position = [float(file_lines[index+10][file_lines[index+10].find(":")+2:-1]),
                    float(file_lines[index+11][file_lines[index+11].find(":")+2:-1]),
                    float(file_lines[index+12][file_lines[index+12].find(":")+2:-1])]
        orientation = [float(file_lines[index+14][file_lines[index+14].find(":")+2:-1]),
                       float(file_lines[index+15][file_lines[index+15].find(":")+2:-1]),
                       float(file_lines[index+16][file_lines[index+16].find(":")+2:-1]),
                       float(file_lines[index+17][file_lines[index+17].find(":")+2:-1])]

        quat = quaternion.from_float_array(orientation)
        odometry_data.append(OdomDatum(sec, nsec, position, quat))

'''
# fake odometry data for a straight path with camera pointed down
odometry_data = []
for index in range(0, 1000):
    quat = quaternion.from_rotation_vector(np.array([1,0,0])*1.5708)
    #quat = quat*quaternion.from_rotation_vector(np.array([0,1,0])*1.5708)
    odometry_data.append(OdomDatum(index, index, [0, index, 700], quat)) # odom is pointed in -y up is z 1.5708 , [1.5708, 1.5708, 0]
'''
### END PARSE ODOM FILE INTO ARRAY AND OBJECT ###


### START READ IN HYPERSPECTRAL FILE ###
hyperspectral_file = open('./hyperspectral.txt', 'r')
lines = hyperspectral_file.readlines()
file_lines = []
for line in lines:
    file_lines.append(line)
### END READ IN HYPERSPECTRAL FILE ###


### START PARSE HYPERSPECTRAL FILE ###
### END PARSE HYPERSPECTRAL FILE ###


### START MATCH HYPERSPECTRAL OBJECT WITH APPROPRIATE ODOM ELEMENT ###
# STEPS
# sort the odom and hyperspectral object arrays by 1.sec and 2.nsec (should be sorted but a couple elements out of oder
# for each element in the hyperspectral array find the closest odom element in t based off of time using binary search
# place odom element index into the coresponding hyperspectral object
### END MATCH HYPERSPECTRAL OBJECT WITH APPROPRIATE ODOM ELEMENT ###


### START DISPLAY PATH IN 3D GRAPH ### 
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')

x = []
y = []
z = []
u = []
v = []
w = []
for datum in odometry_data:
    x.append(datum.position[0])
    y.append(datum.position[1])
    z.append(datum.position[2])

    odom_vector = quaternion.rotate_vectors(datum.orientation, [0, -1, 0]) 
    # print("forward", odom_vector)

    u.append(odom_vector[0])
    v.append(odom_vector[1])
    w.append(odom_vector[2])

ax.quiver(x, y, z, u, v, w, length=1, normalize=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
### END DISPLAY PATH IN 3D GRAPH ###


### START PROJECT PICTURE ONTO ARRAY FROM COLLECTED PATH ###
row_fov = 78.8 # 78.8 degree fov in original photo
row_fov = np.radians(row_fov)
#img = cv2.imread('./Downloads/aerialPhotograph.jpeg', 0)
img = cv2.imread('./checker.jpg', 0)
plt.imshow(img)
plt.show()

casted_img = [[0 for col in range(3000)] for row in range(3000)] # blank canvas for the image to be reprojected onto
for pixel_col_index, datum in enumerate(odometry_data): # for each column in the image
    odom_vector = quaternion.rotate_vectors(datum.orientation, [0, -1, 0])
    axis_of_rotation = quaternion.rotate_vectors(datum.orientation, [0, 0, 1])
    normalized_vector = np.array(axis_of_rotation)/np.linalg.norm(np.array(axis_of_rotation))

    # For each linescan point
    for pixel_row_index in range(img.shape[0]): # for each row in the image
        if pixel_col_index >= img.shape[1]:
            break
        pixel_value = img[pixel_row_index, pixel_col_index]
        raycast_angle = remap_range(pixel_row_index, [0, img.shape[0]-1], [(row_fov/2)-row_fov, row_fov/2])

        quat = quaternion.from_rotation_vector(normalized_vector*raycast_angle) # normalize and multiply
        raycast_vector = quaternion.rotate_vectors(quat, odom_vector)

        end_point = datum.position+(raycast_vector*10000) # length of 10000 from camera
        intersect = isect_line_plane_v3(datum.position, end_point, (0, 0, -200), (0,0,1)) # A plane with a normal of +z and z intercept of -200
        if intersect is None:
            print("ERROR: parallel?")
            continue

        offset_of_model = 700 # this moves the intersections down and to the right to avoid the casted picture running off frame
        intersect = (intersect[0]+offset_of_model, intersect[1]+offset_of_model, intersect[2]+offset_of_model)
        if intersect[0]<len(casted_img) and intersect[1]<len(casted_img[0]) and intersect[0]>=0 and intersect[1]>=0:
            casted_img[int(intersect[0])][int(intersect[1])] = img[pixel_row_index][pixel_col_index]

plt.imshow(casted_img)
plt.show()
### END PROJECT PICTURE ONTO ARRAY FROM COLLECTED PATH ###






