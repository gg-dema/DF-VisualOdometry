from flow_net.flow_net_ui import Flow_net_ui 
from depth_net.depth_net import Depth_net
import numpy as np
import PIL.Image
from torchvision import transforms
from sklearn import linear_model
from tracker.tracker_v3 import TrackerInterface
import pickle
from camera_module import CameraProperty
import torch
from scale_recovery import simple_scale_recovery
import os


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

def main():
    default_camera_calib = [
        [718.856, 0.00000, 607.1928],
        [0.00000, 718.856, 185.2157],
        [0.00000, 0.00000, 1.000000],
    ]

    ground_truth = load_poses_from_txt("data_odometry_poses/dataset/poses/00.txt")
    gt_0 = ground_truth[0]
    # Path to the calib.txt file
    calib_file = 'data_odometry_color/dataset/sequences/00/calib.txt'

    # Read the calib.txt file
    with open(calib_file, 'r') as file:
        calib_data = file.readlines()

    # Extract the camera matrix P0 (or P1, P2, P3) from the calib_data
    p_matrix = calib_data[0].strip().split(' ')[1:]  # Assuming P0 is in the first line

    # Parse the P matrix values
    p_matrix_values = [float(val) for val in p_matrix]

    # Reshape the values into a 3x4 matrix
    p_matrix_array = np.array(p_matrix_values).reshape(3, 4)

    # Extract the intrinsic matrix from the camera matrix
    intrinsic_matrix = p_matrix_array[:, :3]

    print("Intrinsic Matrix:")
    print(intrinsic_matrix)


    cam_prop = CameraProperty(np.array(intrinsic_matrix))

    config_dict = {'robust_E_tracker': False, }
    tracker_ui = TrackerInterface(cam_prop, config_dict)

    flow_ui = Flow_net_ui()
    depNetwork = Depth_net()

    folder_path = 'data_odometry_color/dataset/sequences/00/image_2'
    i = -1
    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):    
        if i == -1:
            path_2 = filename
            i += 1
            continue
            
        i += 1
        curr_gt = ground_truth[i-1]#because gt start from 0, but gt[0] corresponds to the transition from frame 0 to frame 1, with is computed when i = 1
        file_path = os.path.join(folder_path, filename)
        path_1 = path_2
        path_2 = file_path

        #is forward really needed here?
        cols, rows, matched_cols, matched_rows, forward = flow_ui.get_matches(path_1, path_2)

        kp1 = np.array((rows, cols)).T
        kp2 = np.array((matched_rows, matched_cols)).T
        
        pose = tracker_ui.get_pose_from_2d(kp1, kp2)
        R = torch.FloatTensor(pose['R'])
        t = torch.FloatTensor(pose['t'])
        depth = depNetwork.predict("000000.png")
        row_indices_tensor = torch.tensor(rows)
        col_indices_tensor = torch.tensor(cols)

        # Ensure depth map and index tensors are of the same data type
        depth_map_tensor = depth.to(torch.float32)
        row_indices_tensor = row_indices_tensor.to(torch.int64)

        # Compute flattened indices based on row and column indices
        flattened_indices = row_indices_tensor * depth_map_tensor.shape[1] + col_indices_tensor

        # Collect depth values for each point using tensor indexing
        depth_values = depth_map_tensor.view(-1).gather(0, flattened_indices)
        s = simple_scale_recovery(kp1.T, kp2.T, R, t, torch.FloatTensor(cam_prop.intrinsics_matrix), depth_values.detach().numpy())
        st = s*t
        curr_pred = torch.zeros(size=(4,4))
        curr_pred[:3,:3] = R
        curr_pred[:3, 3] = st[0]
        curr_pred[3,3] = 1
        if i == 1: 
            pred_0 = curr_pred
        r_error, t_error = compute_error(gt_0 = gt_0, pred_0 = pred_0.detach().numpy(), curr_gt = curr_gt, curr_pred = curr_pred.detach().numpy())
        print(i, r_error, t_error)

if __name__ == '__main__':
    main()

