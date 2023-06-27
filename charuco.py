import cv2
import numpy as np
import math
import os
import pickle
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def inverseRotation(rvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Transpose the rotation matrix to get the inverse rotation
    R_inv = R.T
    # Convert back to rotation vector
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv


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


def get_rot_str_from_rvec(rvec):
    # Convert the rotation matrix to Euler angles
    pitch, yaw, roll = rotationMatrixToEulerAngles(rvec)
    # Convert the angles from radians to degrees and format them as strings, and invert roll and pitch
    rot_str = f"{-math.degrees(roll)},{math.degrees(pitch)},{math.degrees(yaw)}"
    return rot_str


def restore_position_and_rotation(transform_matrix, secondary_position, secondary_rotation):
    # Convert the secondary position and rotation to homogeneous coordinates
    secondary_position_homogeneous = np.array([
        [secondary_position[0].item()],
        [secondary_position[1].item()],
        [secondary_position[2].item()], 
        [1]
        ])
    # secondary_rotation.shape is (3,3)
    secondary_rotation_homogeneous = np.array([
        [secondary_rotation[0][0].item(), secondary_rotation[0][1].item(), secondary_rotation[0][2].item()],
        [secondary_rotation[1][0].item(), secondary_rotation[1][1].item(), secondary_rotation[1][2].item()],
        [secondary_rotation[2][0].item(), secondary_rotation[2][1].item(), secondary_rotation[2][2].item()],
        [0,0,0]
        ])  # Convert rotation to quaternion    

    # Apply the transformation matrix to the secondary position and rotation
    restored_position_homogeneous = np.dot(transform_matrix, secondary_position_homogeneous)
    restored_rotation_homogeneous = np.dot(transform_matrix[:3, :3], secondary_rotation_homogeneous[:3])  # Apply rotation only

    # Convert the restored position and rotation back to Cartesian coordinates
    # Remove the extra dimension from the restored_position_homogeneous
    restored_position_homo_reshape = np.squeeze(restored_position_homogeneous)
    restored_position = restored_position_homo_reshape[:3]
    restored_rotation = restored_rotation_homogeneous

    # print('secondary_position',secondary_position)
    # print('restored_position',restored_position)
    # print('secondary_rotation',secondary_rotation)
    # print('restored_rotation',restored_rotation)

    return restored_position.reshape(-1, 1), restored_rotation


def rotation_matrix_to_quaternion(rvec):
    r00 = rvec[0, 0]
    r11 = rvec[1, 1]
    r22 = rvec[2, 2]
    r21 = rvec[2, 1]
    r12 = rvec[1, 2]
    r02 = rvec[0, 2]
    r20 = rvec[2, 0]
    r10 = rvec[1, 0]
    r01 = rvec[0, 1]

    q0 = np.sqrt(1 + r00 + r11 + r22) / 2
    if q0 < 1e-6:  # Check if q0 is close to zero
        return np.array([0, 0, 0, 0])  # Return a zero quaternion if q0 is zero

    q1 = (r21 - r12) / (4 * q0)
    q2 = (r02 - r20) / (4 * q0)
    q3 = (r10 - r01) / (4 * q0)

    quaternion = np.array([q0, q1, q2, q3])

    return quaternion


def quaternion_conjugate(quaternion):
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def quaternion_multiply(quaternion1, quaternion2):
    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])


def apply_rotation_difference(quaternion_current, quaternion_difference):
    return quaternion_multiply(quaternion_difference, quaternion_current)


def quaternion_to_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    r00 = 1 - 2 * (q2**2 + q3**2)
    r01 = 2 * (q1*q2 - q0*q3)
    r02 = 2 * (q1*q3 + q0*q2)
    r10 = 2 * (q1*q2 + q0*q3)
    r11 = 1 - 2 * (q1**2 + q3**2)
    r12 = 2 * (q2*q3 - q0*q1)
    r20 = 2 * (q1*q3 - q0*q2)
    r21 = 2 * (q2*q3 + q0*q1)
    r22 = 1 - 2 * (q1**2 + q2**2)
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


def apply_rotation_to_translation(translation_vector, rotation_matrix):
    return np.dot(rotation_matrix, translation_vector)


def calculate_new_position(current_position, translation_vector):
    return np.add(current_position, translation_vector)


def doAruco(directory, fn, K , distCoeff, squareLength, dict_id, cameras, reference_camera_name):

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

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    # Load dictionary
    filename = os.path.join('data', 'dictionaries', "dictionary_"+str(dict_id)+".yaml")
    # Create a FileStorage object to read the dictionary
    fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
    # Read the dictionary from the file
    aruco_dict.readDictionary(fs.getNode("dictionary"))
    # Release the FileStorage object
    fs.release()

    parameters = cv2.aruco_DetectorParameters.create()

    markerLength = 0.8 * squareLength  # Here, our measurement unit is centimetre. #was 0.75
    board = cv2.aruco.CharucoBoard_create(3,3, squareLength, markerLength, aruco_dict)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict,parameters=parameters)  # First, detect markers
    refine_result = cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

    if ids is None:
        # print("No ids found in ", fn, " from dict ", dict_id)
        pass
    else: # At least one marker detected
        primary_charuco_camera_name = str(0)+fn
        charucoretval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray,board)
        im_with_charuco_board = cv2.aruco.drawDetectedCornersCharuco(image_dist, charucoCorners, charucoIds,(0, 255, 0))
        
        # 1. Transformation from Non-reference camera to the Secondary Charuco board
        retval, rvecs_current_board, tvecs_current_board = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, K,distCoeff,np.empty(1),np.empty(1))  # posture estimation from a charuco board
        
        # rvecs_current_board = np.array(rvecs_current)
        # tvecs_current_board = np.array(tvecs_current)
        if retval == True:
            if primary_charuco_camera_name in cameras.keys(): # if primary camera is already detected
                print('[', dict_id, '] Camera name: ', fn)
                tvecs_primary_board = cameras[primary_charuco_camera_name]["tvecs_board"]
                rvecs_primary_board = cameras[primary_charuco_camera_name]["rvecs_board"]
                
                # 2. Transformation from the secondary Charuco board to the Reference camera (first in list)
                # This is the inverse of the transformation from the Reference camera to the secondary Charuco board, 
                # which you can calculate using the cv2.estimatePoseCharucoBoard function and then invert. 
                # This will give the rotation and translation vectors 
                # (rvecs_secondary_to_A and tvecs_secondary_to_A) from the secondary Charuco board 
                # to camera A.
                # reference_camera_name = str(dict_id)+fn # Secondary board is dict_id
                print('cameras.keys()', cameras.keys())
                # tvecs_reference_camera = cameras[reference_camera_name]["tvecs_camera"]
                # rvecs_reference_camera = cameras[reference_camera_name]["rvecs_camera"]
                # print('ref camera name:',cameras[reference_camera_name]['name'])

                """print('tvecs_current_board: ', tvecs_current_board)
                print('tvecs_primary_board: ', tvecs_primary_board)
                print('rvecs_current_board: ', rvecs_current_board)
                print('rvecs_primary_board: ', rvecs_primary_board)"""
                
                # Get the difference between the primary board and the current board
                """tvecs_board_difference = tvecs_primary_board - tvecs_current_board # emulation
                rvecs_board_difference = rvecs_primary_board - rvecs_current_board # emulation
                print('tvecs_board_difference: ', tvecs_board_difference)
                print('rvecs_board_difference: ', rvecs_board_difference)
                rvecs_current_board = rvecs_current_board + rvecs_board_difference
                tvecs_current_board = tvecs_current_board + tvecs_board_difference"""

            im_with_charuco_board = cv2.drawFrameAxes(
                im_with_charuco_board, 
                K, 
                distCoeff, 
                rvecs_current_board, 
                tvecs_current_board,
                100 # axis length 100 can be changed according to your requirement
                )
            
            # str_position = "CAMERA Position x=%4.1f  y=%4.1f z=%4.1f" % (-pos_camera[2], pos_camera[0], pos_camera[1])
            # -- Get the attitude of the camera respect to the frame
            # pitch_camera, yaw_camera, roll_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
            dict_path = os.path.join(directory, str(dict_id))
            if not os.path.exists(dict_path):
                os.makedirs(dict_path)

            R_flip  = np.zeros((3,3), dtype=np.float32)
            R_flip[0,0] = 1.0
            R_flip[1,1] =-1.0
            R_flip[2,2] =-1.0

            # -- Now get Position and attitude f the camera respect to the marker
            R_ct = np.matrix(cv2.Rodrigues(rvecs_current_board)[0])
            R_tc = R_ct.T
            tvecs_current_camera = -R_tc * np.matrix(tvecs_current_board)

            """if primary_charuco_camera_name in cameras.keys(): # if primary camera is already detected
                tvecs_primary_board = cameras[primary_charuco_camera_name]["tvecs_board"]
                rvecs_primary_board = cameras[primary_charuco_camera_name]["rvecs_board"]
                print('tvecs_current_board: ', tvecs_current_board)
                print('tvecs_primary_board: ', tvecs_primary_board)
                print('rvecs_current_board: ', rvecs_current_board)
                print('rvecs_primary_board: ', rvecs_primary_board)                
                # Convert rotation vectors to rotation matrices
                R_current_board, _ = cv2.Rodrigues(rvecs_current_board)
                R_primary_board, _ = cv2.Rodrigues(rvecs_primary_board)
                # Compute the rotation matrix that transforms points from the current board's frame to the primary board's frame
                R = np.matmul(R_primary_board, R_current_board.T)
                # Compute the translation vector that transforms points from the current board's frame to the primary board's frame
                tvecs_current = tvecs_primary_board - np.matmul(R, tvecs_current_board)
                # Convert the rotation matrix back to a rotation vector
                rvecs_current, _ = cv2.Rodrigues(R)
                print('Board tvecs were updated from the primary board')"""

            """if primary_charuco_camera_name in cameras.keys(): # if primary camera is already detected
                # Position and rotation correction because of secondary board is shifted and rotated
                primary_charucoCorners = cameras[primary_charuco_camera_name]['charucoCorners']
                current_charucoCorners = charucoCorners
                rvecs_primary = cameras[primary_charuco_camera_name]['rvecs']
                tvecs_primary = cameras[primary_charuco_camera_name]['tvecs']
                tvecs_diff = tvecs_primary - tvecs_current              
                # camera_pos_difference = tvecs_primary - tvecs_current
                formatted_difference = np.array2string(tvecs_diff, precision=2, suppress_small=True)
                
                print('\ncharuco id: ', dict_id)
                print('camera name: ', fn)
                
                print('primary_charucoCorners: ', primary_charucoCorners)
                print('current_charucoCorners: ', current_charucoCorners)
                
                print('primary_camera_pos: ', tvecs_primary)
                print('tvecs_current: ', tvecs_current)
                print('position difference:', formatted_difference)
                
                print('rvect_primary: ', rvecs_primary)
                print('rvecs_current: ', rvecs_current)
                
                # tvecs_current = tvecs_current + tvecs_diff # emulation
                print('New tvecs_current: ', tvecs_current)
                
                # rvecs_diff = rvecs_primary - rvecs_current
                # rvecs_current = rvecs_diff + rvecs_current # emulation
                print('New rvecs_current: ', rvecs_current)"""
            
            # Save the str_position and str_attitude
            pos_str = str(tvecs_current_camera[0,0])+','+str(tvecs_current_camera[1,0])+','+str(-tvecs_current_camera[2,0])
            # Rotation
            rvec_matrix, _ = cv2.Rodrigues(rvecs_current_board)
            # rot_str = str(math.degrees(roll_camera)) + ',' + str(math.degrees(pitch_camera)) + ',' + str(math.degrees(yaw_camera))
            rvec = get_rvec_for_plotting(rvecs_current_board)
            rot_str = get_rot_str_from_rvec(rvec)
            
            cam_txt_path = os.path.join(dict_path, 'cameras')
            if not os.path.exists(cam_txt_path):
                os.makedirs(cam_txt_path)

            with open(os.path.join(cam_txt_path, fn + '.txt'), 'w') as f:
                f.write(pos_str)
                f.write('\n' + rot_str)

            with open(os.path.join(dict_path, fn + '.R'), 'wb') as f:
                pickle.dump(rvec_matrix, f)
            with open(os.path.join(dict_path, fn + '.T'), 'wb') as f:
                pickle.dump(tvecs_current_camera, f)
            with open(os.path.join(dict_path, fn + '.K'), 'wb') as f:
                pickle.dump(K, f)

            # Save im_with_charuco_board to a charucoDiagnostics folder
            cD_path = os.path.join(dict_path, 'charucoDiagnostics')
            if not os.path.exists(cD_path):
                os.makedirs(cD_path)
            cv2.imwrite(os.path.join(cD_path, fn), im_with_charuco_board)
            
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

    
    # return None, None, None, None, None
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
    ax.set_title("Camera Positions")
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

        # Plot the direction line
        # print("camera['t_vecs'][0]", camera['t_vecs'][0])
        # print('x_end', x_end)
        ax.plot(
            [tvec[0], x_end], 
            [tvec[1], y_end], 
            [tvec[2], z_end], 
            color=camera['color']
            )


def main():
    
    charuco_targets_count = 6
    # squareLength = 30.48  # charuco size 30.48cm = 36 inch target, 33.86 = 40 inch target, 20.319 = 24 inch target
    
    K, distCoeff = getCameraMatrix(CAMERATYPE = 3)

    print('opencv-python version:', cv2.__version__)

    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="This script requires a directory path where images are stored. "
                                                "Usage: python your_script.py /path/to/your/images/"
        )
    # Add the directory argument
    # parser.add_argument("--directory", help="Directory of the images", default="data/datasets/HD100ALow_twin_bottom/charuco")
    parser.add_argument("--directory", help="Directory of the images")
    # Parse the arguments
    args = parser.parse_args()
    # If the user doesn't provide any arguments, print the help message and exit
    if not vars(args):
        parser.print_help()
        parser.exit(1)
    directory = args.directory

    cameras = dict()
    print('directory:', directory)
    reference_camera_name = ''
    for dict_id in range(charuco_targets_count):
        squareLength = get_squareLength(dict_id)
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                if reference_camera_name == '':
                    reference_camera_name = filename
                # if filename != 'RightB.png':
                # if filename != 'RightTop.png':
                #     continue
                # print('# calling: ', filename, 'squareLength:', squareLength, 'dict_id:', dict_id)
                # rvecs_current, tvecs_current, charucoCorners, rvecs_board, tvecs_board = doAruco(
                cameras = doAruco(
                    directory,
                    filename, 
                    K , 
                    distCoeff, 
                    squareLength,
                    dict_id,
                    cameras,
                    reference_camera_name
                    )
                # if rvecs_current is None or tvecs_current is None or charucoCorners is None or tvecs_board is None or rvecs_board is None:
                #     pass
                # else:
                
                # Add charuco camera
                # if rvec is not None and tvec is not None:
                """cameras[str(dict_id)+filename] = {
                    'rvecs':rvecs_current,
                    'tvecs':tvecs_current,
                    'charucoCorners':charucoCorners,
                    'rvecs_board':rvecs_board,
                    'tvecs_board':tvecs_board,
                    'color':colors[dict_id],
                    'marker':markers[dict_id],
                }"""
                # print('[', dict_id, ']', filename, tvec)
                # print(f"[ {dict_id} ] {filename} [{', '.join(f'{x:.0f}' for x in tvec)}]")
            else:
                # Not image file
                continue

    if len(cameras) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # print(cameras)
        plot_charuco_camera_directions(ax, cameras, 400)
        plot_cameras(ax, cameras)
    
    print('Done')


if __name__ == '__main__':
    main()
