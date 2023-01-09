
casted_img = [[0 for col in range(3000)] for row in range(3000)] # blank canvas for the image to be reprojected onto
wavelength_index = 176
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
