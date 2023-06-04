import numpy as np
import math
from camera_module import CameraProperty
import matplotlib.pyplot as plt
import matplotlib as mpl


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def plot_trajectory(poses_gt, poses_result, seq):
        """Plot trajectory for both GT and prediction
        
        Args:
            poses_gt (dict): {idx: 4x4 array}; ground truth poses
            poses_result (dict): {idx: 4x4 array}; predicted poses
            seq (int): sequence index.
        """
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        color_list = {"Ground Truth": 'k',
                      "Ours": 'lime'}
        linestyle = {"Ground Truth": "--",
                     "Ours": "-"}

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xyz = []
            frame_idx_list = sorted(poses_dict["Ours"].keys())
            for frame_idx in frame_idx_list:
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = poses_dict[key][frame_idx]
                pos_xyz.append([pose[0, 3], pose[1, 3], pose[2, 3]])
            pos_xyz = np.asarray(pos_xyz)
            plt.plot(pos_xyz[:, 0],  pos_xyz[:, 2], label=key, c=color_list[key], linestyle=linestyle[key])

            # Draw rect
            if key == 'Ground Truth':
                rect = mpl.patches.Rectangle((pos_xyz[0, 0]-5, pos_xyz[0, 2]-5), 10,10, linewidth=2, edgecolor='k', facecolor='none')
                ax.add_patch(rect)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        plt.grid(linestyle="--")
        fig.set_size_inches(10, 10)
        png_title = "sequence_{}".format(seq)
        fig_pdf = png_title + "test_loro.pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def load_poses_from_txt(file_name):
    """Load poses from txt (KITTI format)
    Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)

    Args:
        file_name (str): txt file path
    
    Returns:
        poses (dict): {idx: [4x4] array}
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i!=""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses


def rotation_error(pose_error):
    """Compute rotation error
    
    Args:
        pose_error (array, [4x4]): relative pose error
    
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    
    Args:
        pose_error (array, [4x4]): relative pose error
    
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error


def compute_error(gt_0, pred_0, curr_gt, curr_pred):
    gt1 = gt_0
    gt2 = curr_gt
    gt_rel = np.linalg.inv(gt1) @ gt2

    pred1 = pred_0
    pred2 = curr_pred
    pred_rel = np.linalg.inv(pred1) @ pred2
    rel_err = np.linalg.inv(gt_rel) @ pred_rel
    
    t_error = translation_error(rel_err)
    r_error = rotation_error(rel_err)
    return r_error, t_error

def cam_stuff(seq):
        # Path to the calib.txt file
    calib_file = 'data_odometry_color/dataset/sequences/'+seq+'/calib.txt'

    # Read the calib.txt file
    with open(calib_file, 'r') as file:
        calib_data = file.readlines()

    # Extract the camera matrix P0 (or P1, P2, P3) from the calib_data
    p_matrix = calib_data[2].strip().split(' ')[1:]  # Assuming P0 is in the first line

    # Parse the P matrix values
    p_matrix_values = [float(val) for val in p_matrix]

    # Reshape the values into a 3x4 matrix
    p_matrix_array = np.array(p_matrix_values).reshape(3, 4)

    # Extract the intrinsic matrix from the camera matrix
    intrinsic_matrix = p_matrix_array[:, :3]

    print("Intrinsic Matrix:")
    print(intrinsic_matrix)
    cam_prop = CameraProperty(np.array(intrinsic_matrix))

    
    config_dict = {'robust_tracker': False,
                   'numb_robust_iter': 10}
    return cam_prop, config_dict