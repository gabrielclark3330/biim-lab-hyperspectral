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

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# remaps a value in range r1 into a returned value in range r2
def remap_range(value, r1, r2): # ranges are inclusive on both sides
    return r2[0] + (value - r1[0]) * (r2[1] - r2[0]) / (r1[1] - r1[0])

class OdomDatum():
    # position in [x,y,z] orientation in [x,y,z,w]
    def __init__(self, sec, nsec, position, orientation, euler_angles):
        self.sec = sec
        self.nsec = nsec
        self.position = position
        self.orientation = orientation 
        self.euler_angles = euler_angles
### END HELPER FUNCTIONS ###


### START READ IN FILE ###
odom_file = open('./odom.txt', 'r')
lines = odom_file.readlines()
file_lines = []
for line in lines:
    file_lines.append(line)


'''Example of repeating data in file ---->>>
---
header: 
  seq: 962
  stamp: 
    secs: 1670975229
    nsecs: 240288258
  frame_id: "odom"
child_frame_id: "camera_link"
pose: 
  pose: 
    position: 
      x: 0.07615969330072403
      y: -0.13087941706180573
      z: 0.02420314960181713
    orientation: 
      x: -0.11165079065823946
      y: -0.08316018677949848
      z: -0.6891438953198216
      w: 0.7111252884152074
  covariance: [0.0003328895030985587, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003328895030985587, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003328895030985587, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00037179893299005935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00037179893299005935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00037179893299005935]
twist: 
  twist: 
    linear: 
      x: 0.02463964931666851
      y: 0.017370542511343956
      z: 0.007171323522925377
    angular: 
      x: -0.014589725993573666
      y: 0.011161587201058865
      z: -0.003960487898439169
  covariance: [0.00016644475154927935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00016644475154927935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00016644475154927935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00018589946649502967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00018589946649502967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00018589946649502967]
---'''
### END READ IN FILE ###


### START PARSE FILE INTO ARRAY AND OBJECT ###
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
        euler_angles = euler_from_quaternion(orientation[0], 
                                            orientation[1], 
                                            orientation[2], 
                                            orientation[3])
        
        # Adjusting rotation of original path for best projection
        euler_angles = (euler_angles[0], euler_angles[1]+1.5708, euler_angles[2]) # +4.71239 = 270deg

        odometry_data.append(OdomDatum(sec, nsec, position, orientation, euler_angles))

odometry_data = []
for index in range(0, 20):
    odometry_data.append(OdomDatum(index, index, [0, index, 700], [0,0,0,0], [1.5708, 1.5708, 0])) # odom is pointed in -y up is z 1.5708
'''
'''
### END PARSE FILE INTO ARRAY AND OBJECT ###


### START DISPLAY PATH IN 3D GRAPH ### 
#NOTE: Y axis is kind of broken but only in the visualizaiton?
#NOTE: Gimble lock is the problem switch to quaternions
fig = plt.figure()
ax = fig.gca(projection='3d')

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

    roll_x, pitch_y, yaw_z = datum.euler_angles # radians
    rotation_mat = np.linalg.multi_dot([rotation_matrix([1,0,0], roll_x),
                                        rotation_matrix([0,1,0], pitch_y),
                                        rotation_matrix([0,0,1], yaw_z)])
    print("forward", np.dot(rotation_mat, [0, -1, 0]))
    '''
    print("up", np.dot(rotation_mat, [0, 0, 1]))
    print("right", np.dot(rotation_mat, [-1, 0, 0]))
    '''
    odom_vector = np.dot(rotation_mat, [0, -1, 0])

    u.append(odom_vector[0])
    v.append(odom_vector[1])
    w.append(odom_vector[2])

ax.quiver(x, y, z, u, v, w, length=10, normalize=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
### END DISPLAY PATH IN 3D GRAPH ###


### START PROJECT PICTURE ONTO ARRAY FROM COLLECTED PATH ###
row_fov = 78.8 # 78.8 degree fov in original photo
#img = cv2.imread('./Downloads/aerialPhotograph.jpeg', 0)
img = cv2.imread('./Downloads/checker.jpg', 0)
plt.imshow(img)
plt.show()
casted_img = [[0 for col in range(2000)] for row in range(2000)]
for pixel_col_index, datum in enumerate(odometry_data):
    roll_x, pitch_y, yaw_z = datum.euler_angles # radians
    # [0, -1, 0] is forwards on the camera
    # [0, 0, 1] is up on the camera
    # just rotate each of these respectively by the rpy to find where camera is facing
    # and where up is on the camera (the axis of rotation for each point on the linescan)
    rotation_mat = np.linalg.multi_dot([rotation_matrix([1,0,0], roll_x),
                                           rotation_matrix([0,1,0], pitch_y),
                                           rotation_matrix([0,0,1], yaw_z)])
    odom_vector = np.dot(rotation_mat, [0, -1, 0])
    axis_of_rotation = np.dot(rotation_mat, [0, 0, 1])

    # For each linescan point
    for pixel_row_index in range(img.shape[0]): # for each row
        if pixel_col_index >= img.shape[1]:
            break
        pixel_value = img[pixel_row_index, pixel_col_index]
        raycast_angle = remap_range(pixel_row_index, [0, img.shape[0]-1], [(row_fov/2)-row_fov, row_fov/2])
        raycast_angle = np.radians(raycast_angle)

        rotation_mat = rotation_matrix(axis_of_rotation, raycast_angle)
        raycast_vector = np.dot(rotation_mat, odom_vector)
        end_point = datum.position+(raycast_vector*10000) # length of 10000 from camera
        intersect = isect_line_plane_v3(datum.position, end_point, (0, 0, -200), (0,0,1)) # A plane with a normal of +z and z intercept of -200
        if intersect is None:
            print("ERROR: parallel?")
            continue

        offset_of_model = 500 # this moves the intersections down and to the right to avoid the casted picture running off frame
        intersect = (intersect[0]+offset_of_model, intersect[1]+offset_of_model, intersect[2]+offset_of_model)
        if intersect[0]<len(casted_img) and intersect[1]<len(casted_img[0]) and intersect[0]>=0 and intersect[1]>=0:
            casted_img[int(intersect[0])][int(intersect[1])] = img[pixel_row_index][pixel_col_index]

plt.imshow(casted_img)
plt.show()
### END PROJECT PICTURE ONTO ARRAY FROM COLLECTED PATH ###






