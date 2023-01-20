import math
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import cv2
import quaternion
import pickle

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


def remap_range(value, r1, r2):
    """
    Ranges are inclusive on both sides
    Remaps a value in range r1 into a returned value in range r2
    """
    return r2[0] + (value - r1[0]) * (r2[1] - r2[0]) / (r1[1] - r1[0])

def interpolate_points_from_times(xyz1, xyz2, t1, t2, t_out):
    """
    xyz1, xyz2: Arrays representing two points in 3d space
    t1, t2, t_out: Integers representing times
        t1, t2: Represent a span of time
        t_out: A time in relation to t1 and t2
    
    Returns an array representing a point in 3d space between xyz1 and xyz2 that is proportionally equal to the amount t_out is between t1 and t2.
    """
    return [
            remap_range(t_out, [t1, t2], [xyz1[0], xyz2[0]]),
            remap_range(t_out, [t1, t2], [xyz1[1], xyz2[1]]),
            remap_range(t_out, [t1, t2], [xyz1[2], xyz2[2]])]

def binary_search(arr, x):
    """
    returns index of the closest element in arr that is not bigger than x
    """
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:

        mid = (high + low) // 2

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1

        # means x is present at mid
        else:
            return mid
    return mid

class OdomDatum():
    # position in [x,y,z] orientation in [x,y,z,w]
    def __init__(self, sec, nsec, position, orientation):
        self.sec = sec
        self.nsec = nsec
        self.position = position
        self.orientation = orientation 

class HYSPMDatum():
    # linescan is a matrix where each column is a line scan and each row as a different frequency
    def __init__(self, sec, nsec, line_scan, odom_match_index=None):
        self.sec = sec
        self.nsec = nsec
        self.line_scan = line_scan
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
# hyperspectral_data is formated as an array of 2d arrays where each 2d arrays columns contains a linescan and each row is that linescan at different wavelengths 
hyperspectral_data = []
with open("hyperspectralData.pick", "rb") as f:
    hyperspectral_data = pickle.load(f)
# hyperspectral_timestamps is formated as an array of tuples where the tuple at index i contains (sec, nsec) for the ith hyperspectral_datum
hyperspectral_timestamps = []
with open("hyperspectralTimestamps.pick", "rb") as f:
    hyperspectral_timestamps = pickle.load(f)

hyperspectral_objs = []
for i in range(len(hyperspectral_data)):
    hyperspectral_objs.append(HYSPMDatum(hyperspectral_timestamps[i][0], hyperspectral_timestamps[i][1], hyperspectral_data[i]))
### END READ IN HYPERSPECTRAL FILE ###


### START DISPLAY UNWARPED HYPERSPECTRAL IMAGE ON 1 WAVELEN
# Unedited hyperspectral data displayed
hyper_cube = np.array(hyperspectral_data)
display = hyper_cube[:,:,176]
plt.imshow(display)
plt.show()
### END DISPLAY UNWARPED HYPERSPECTRAL IMAGE ON 1 WAVELEN


### START MATCH HYPERSPECTRAL OBJECT WITH APPROPRIATE ODOM ELEMENT ###
# sort the odom and hyperspectral object arrays first by sec and then nsec to break ties (should be sorted but a couple elements out of oder
odometry_data.sort(key=lambda x: (x.sec, x.nsec))
hyperspectral_objs.sort(key=lambda x: (x.sec, x.nsec))
# for each element in the hyperspectral array find the closest odom element in t based off of time using binary search
hash_arr = [x.sec*(10**9)+x.nsec for x in odometry_data]
for hyspim_obj in hyperspectral_objs:
    index = binary_search(hash_arr, hyspim_obj.sec*(10**9)+hyspim_obj.nsec)
    hyspim_obj.odom_match_index = index
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


### 
def find_odom_objs_surrounding_time(odom_objs, time):
    hyspim_obj.sec*(10**9)+hyspim_obj.nsec
    hyspim_obj.odom_match_index
    return odom_objs
for hyspim_obj in hyperspectral_objs:
    hyspim_obj_between_odom_objs = find_odom_objs_surrounding_time(odom_objs, time)
    interpolated_odom_point = interpolate_points_from_times(xyz1, xyz2, t1, t2, t_out)
    interpolated_odom_quat = quaternion.slerp(R1, R2, t1, t2, t_out)

###


### START PROJECT PICTURE ONTO ARRAY FROM COLLECTED PATH ###
#img = cv2.imread('./Downloads/aerialPhotograph.jpeg', 0)
#img = cv2.imread('./checker.jpg', 0)
#plt.imshow(img)
#plt.show()

casted_img = [[0 for col in range(3000)] for row in range(3000)] # blank canvas for the image to be reprojected onto
wavelength_index = 176
row_fov = np.radians(78.8) # 78.8 degree fov in original photo
for hyspim_obj in hyperspectral_objs:
    odometry_datum = odometry_data[hyspim_obj.odom_match_index]
    odom_vector = quaternion.rotate_vectors(odometry_datum.orientation, [0, -1, 0])
    axis_of_rotation = quaternion.rotate_vectors(odometry_datum.orientation, [0, 0, 1])
    normalized_vector = np.array(axis_of_rotation)/np.linalg.norm(np.array(axis_of_rotation))

    for pixel_row_index in range(len(hyspim_obj.line_scan)):
        raycast_angle = remap_range(pixel_row_index, [0, len(hyspim_obj.linescan)-1], [(row_fov/2)-row_fov, row_fov/2])

        quat = quaternion.from_rotation_vector(normalized_vector*raycast_angle) # normalize and multiply
        raycast_vector = quaternion.rotate_vectors(quat, odom_vector)

        end_point = odometry_datum.position+(raycast_vector*10000) # length of 10000 from camera
        intersect = isect_line_plane_v3(odometry_datum.position, end_point, (0,0,-200), (0,0,1)) # A plane with a normal of +z and z intercept of -200
        if intersect is None:
            print("ERROR: parallel?")
            continue

        offset_of_model = 700 # this moves the intersections down and to the right to avoid the casted picture running off frame
        intersect = (intersect[0]+offset_of_model, intersect[1]+offset_of_model, intersect[2]+offset_of_model)
        if intersect[0]<len(casted_img) and intersect[1]<len(casted_img[0]) and intersect[0]>=0 and intersect[1]>=0:
            casted_img[int(intersect[0])][int(intersect[1])] = hyspim_obj.line_scan[pixel_row_index][wavelength_index]

plt.imshow(casted_img)
plt.show()
### END PROJECT PICTURE ONTO ARRAY FROM COLLECTED PATH ###






