import cv2
import numpy as np
import math
import os
# import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_squareLength(charuco_id):
    if charuco_id == 0:
        return 30.48 # 36 inches
    elif charuco_id < 10:
        return 10.16 # 12 inches
    elif charuco_id < 20:
        return 13.55 # 16 inches
    elif charuco_id < 30:
        return 20.319 # 24 inches
    elif charuco_id < 40:
        return 6.77 # 8 inches
    else:
        return None


def getCameraMatrix(CAMERATYPE):
    K, distCoeff = None, None

    if CAMERATYPE==0: #Blackmagic MFT 14mm lens (Daz)
        distCoeff = np.array([[ -8.88697267e-05 , -2.16339304e-04, 8.87302780e-06,3.45972506e-05,8.01520895e-04]])
        K = np.array(
        [[2.68799464e+03, 0.00000000e+00, 1.07940299e+03],
    [0.00000000e+00, 2.68805782e+03, 1.91945203e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    if CAMERATYPE==1: #Blackmagic MFT 14mm lens (Real Lens)
        distCoeff = np.array([[ -8.88697267e-05 , -2.16339304e-04, 8.87302780e-06,3.45972506e-05,8.01520895e-04]])
        K = np.array(
        [[2.68799464e+03, 0.00000000e+00, 1.07940299e+03],
    [0.00000000e+00, 2.68805782e+03, 1.91945203e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    if CAMERATYPE==2: #HD100A 4mm lens Daz
        distCoeff = np.array([[ 7.94732062e-04 , -6.53440981e-03, 1.59511692e-05,1.34030321e-05,1.38527246e-02]])
        K = np.array(
        [[1.15192133e+03, 0.00000000e+00, 5.39560767e+02],
    [0.00000000e+00, 1.15192586e+03, 9.59589490e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    if CAMERATYPE==3: #HD100A 2.8mm lens Daz taken from real world hd100a
        distCoeff = np.array([[-2.67958291e-05 , -1.68184836e-04, -3.40661391e-05,-3.98153416e-05,5.99942106e-04]])
        K = np.array(
        [[1.05625057e+03, 0.00000000e+00, 5.39346239e+02],
    [0.00000000e+00, 1.05615221e+03, 9.59171972e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
    return K, distCoeff


def get_rvec_for_plotting(rvecs):
    rvec = inverseRotation(rvecs)
    # flatten the 2D array to 1D
    rvec = rvec.flatten()
    # convert numpy floats to python floats
    rvec = [float(x) for x in rvec]
    r_vecs_array = np.array(rvec)
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(r_vecs_array)
    # Flip z-axis: x, y, -z
    flip_yz_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    R_flipped = np.matmul(flip_yz_matrix, R)
    # Convert the new rotation matrix back to a rotation vector
    rvec = R_flipped
    return rvec


def inverseRotation(rvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Transpose the rotation matrix to get the inverse rotation
    R_inv = R.T
    # Convert back to rotation vector
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv


def get_rot_str_from_rvec(rvec):
    # Convert the rotation matrix to Euler angles
    pitch, yaw, roll = rotationMatrixToEulerAngles(rvec)
    # Convert the angles from radians to degrees and format them as strings, and invert roll and pitch
    rot_str = f"{-math.degrees(roll)},{math.degrees(pitch)},{math.degrees(yaw)}"
    return rot_str


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-2

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def load_aruco_dict(dict_id):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    # Load dictionary
    filename = os.path.join('dictionaries', "dictionary_"+str(dict_id)+".yaml")
    # Create a FileStorage object to read the dictionary
    fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
    # Read the dictionary from the file
    aruco_dict.readDictionary(fs.getNode("dictionary"))
    # Release the FileStorage object
    fs.release()
    return aruco_dict

def save_camera_data(dataset, fn, K, rvec_matrix, tvecs_current_camera, pos_str, rot_str):
    cam_txt_path = os.path.join('out', dataset)
    if not os.path.exists(cam_txt_path):
        os.makedirs(cam_txt_path)

    with open(os.path.join(cam_txt_path, fn + '.txt'), 'w') as f:
        f.write(pos_str)
        f.write('\n' + rot_str)

    """with open(os.path.join(dataset, fn + '.R'), 'wb') as f:
        pickle.dump(rvec_matrix, f)
    with open(os.path.join(dataset, fn + '.T'), 'wb') as f:
        pickle.dump(tvecs_current_camera, f)
    with open(os.path.join(dataset, fn + '.K'), 'wb') as f:
        pickle.dump(K, f)"""


def doAruco(directory, dataset, fn, K , distCoeff, squareLength, dict_id, cameras):

    img_dir = os.path.join(directory, fn)
    img = cv2.imread(img_dir)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                K, 
                distCoeff, 
                (img.shape[1], img.shape[0]), 
                1, 
                (img.shape[1], img.shape[0])
                )

    image_dist = cv2.undistort(img, K, distCoeff, None, newcameramtx)

    gray = cv2.cvtColor(image_dist, cv2.COLOR_BGR2GRAY)
    # Apply adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    aruco_dict = load_aruco_dict(dict_id)

    parameters = cv2.aruco_DetectorParameters.create()

    markerLength = 0.8 * squareLength # Here, our measurement unit is centimetre. #was 0.75
    board = cv2.aruco.CharucoBoard_create(3,3, squareLength, markerLength, aruco_dict)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict,parameters=parameters)  # First, detect markers
    refine_result = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

    if ids is None:
        print('[', dict_id, ']', img_dir)
        pass
    else: # At least one marker detected
        print('[', dict_id, ']', img_dir, '*')
        primary_charuco_camera_name = str(0)+fn
        charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray,board)
        im_with_charuco_board = cv2.aruco.drawDetectedCornersCharuco(image_dist, charucoCorners, charucoIds,(0, 255, 0))
        
        # 1. Transformation from Non-reference camera to the Secondary Charuco board
        retval, rvecs_current_board, tvecs_current_board = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, K,distCoeff,np.empty(1),np.empty(1))  # posture estimation from a charuco board
        
        if retval == True:
            if primary_charuco_camera_name in cameras.keys(): # if primary camera is already detected
                print('[', dict_id, '] Camera name: ', fn)
                tvecs_primary_board = cameras[primary_charuco_camera_name]["tvecs_board"]
                rvecs_primary_board = cameras[primary_charuco_camera_name]["rvecs_board"]

            im_with_charuco_board = cv2.drawFrameAxes(
                im_with_charuco_board, 
                K, 
                distCoeff, 
                rvecs_current_board, 
                tvecs_current_board,
                100 # axis length 100 can be changed according to your requirement
                )
            
            R_flip  = np.zeros((3,3), dtype=np.float32)
            R_flip[0,0] = 1.0
            R_flip[1,1] =-1.0
            R_flip[2,2] =-1.0

            # -- Now get Position and attitude f the camera respect to the marker
            R_ct = np.matrix(cv2.Rodrigues(rvecs_current_board)[0])
            R_tc = R_ct.T
            tvecs_current_camera = -R_tc * np.matrix(tvecs_current_board)
            
            # Save the str_position and str_attitude
            pos_str = str(tvecs_current_camera[0,0])+','+str(tvecs_current_camera[1,0])+','+str(-tvecs_current_camera[2,0])
            # Rotation
            rvec_matrix, _ = cv2.Rodrigues(rvecs_current_board)
            # rot_str = str(math.degrees(roll_camera)) + ',' + str(math.degrees(pitch_camera)) + ',' + str(math.degrees(yaw_camera))
            rvec = get_rvec_for_plotting(rvecs_current_board)
            rot_str = get_rot_str_from_rvec(rvec)
            
            save_camera_data(dataset, fn, K, rvec_matrix, tvecs_current_camera, pos_str, rot_str)

            # Save im_with_charuco_board to a charucoDiagnostics folder
            cD_path = 'diagnostics'
            if not os.path.exists(cD_path):
                os.makedirs(cD_path)
            cD_path = os.path.join(cD_path, dataset)
            if not os.path.exists(cD_path):
                os.makedirs(cD_path)            
            cv2.imwrite(os.path.join(cD_path, str(dict_id)+fn), im_with_charuco_board)
            
            # return rvecs_current, tvecs_current, charucoCorners, rvecs_current_board, tvecs_current_board
            colors = [
                'red', 
                'green', 
                'blue', 
                'yellow', 
                'orange', 
                'purple', 
                'pink',
                'cyan'
                ]
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*']
            cameras[str(dict_id)+fn] = {
                        'rvecs':rvecs_current_board,
                        'tvecs':tvecs_current_camera,
                        'charucoCorners':charucoCorners,
                        'rvecs_board':rvecs_current_board,
                        'tvecs_board':tvecs_current_board,
                        'color':colors[dict_id],
                        'marker':markers[dict_id],
                    }

    return cameras


def plot_cameras(ax, cameras_c):

    max_value = 0
    min_value = 0
    for camera_name, camera in cameras_c.items():
        tvecs_current = camera['tvecs']
        tvec = [tvecs_current[0,0], tvecs_current[1,0], -tvecs_current[2,0]]
        x = tvec[0]
        y = tvec[1]
        z = tvec[2]
        ax.scatter(
            x, y, z, 
            c=camera['color'], 
            marker=camera['marker'], 
            label=camera_name
            )
        ax.text(x, y, z, camera_name, fontsize=8)
        max_value = max(max_value, x, y, z)
        min_value = min(min_value, x, y, z)

    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_zlim(min_value, max_value)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plot_charuco_camera_directions(ax, cameras_c, length=100):
    for camera_name, camera in cameras_c.items():

        rvec = get_rvec_for_plotting(camera['rvecs'])
        direction = np.matmul(rvec, np.array([[0], [0], [1]]))

        tvecs_current = camera['tvecs']
        tvec = [tvecs_current[0,0], tvecs_current[1,0], -tvecs_current[2,0]]
        
        # Scale the direction by the desired length
        direction = direction * length

        # Calculate the end point of the direction line
        x_end = tvec[0] + direction[0][0]
        y_end = tvec[1] + direction[1][0]
        z_end = tvec[2] + direction[2][0]

        ax.plot(
            [tvec[0], x_end], 
            [tvec[1], y_end], 
            [tvec[2], z_end], 
            color=camera['color']
            )


def main():
    
    charuco_targets_count = 6
    
    K, distCoeff = getCameraMatrix(CAMERATYPE = 3)

    print('opencv-python version:', cv2.__version__)

    for dataset in range(3):
        print('\ndataset: ', dataset)
        directory = os.path.join('renders', str(dataset))
        cameras = dict()
        for filename in sorted(os.listdir(directory)):
            for dict_id in range(charuco_targets_count):
                squareLength = get_squareLength(dict_id)
                cameras = doAruco(
                    directory,
                    str(dataset),
                    filename, 
                    K , 
                    distCoeff, 
                    squareLength,
                    dict_id,
                    cameras
                    )

        if len(cameras) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Cameras of dataset " + str(dataset))
            plot_charuco_camera_directions(ax, cameras, 400)
            plot_cameras(ax, cameras)
    
    print('Done')


if __name__ == '__main__':
    main()
