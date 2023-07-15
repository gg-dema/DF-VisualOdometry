import numpy as np
import math
from camera_module import CameraProperty
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

def plot_trajectory_drone(poses_gt, poses_preds, output):
        """Plot trajectory for both GT and prediction
        """
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        pose_xy = []
        for pose in poses_gt:
            pose_xy.append(pose)
        pose_xy = np.array(pose_xy)
        plt.plot(pose_xy[:, 0],  -pose_xy[:, 1], label='Ground Truth', c='k', linestyle='--')

        pose_xyz = []
        for pose in poses_preds:
                pose_xyz.append([pose[0, 3], pose[1, 3], pose[2, 3]])
        pose_xyz = np.array(pose_xyz)
        plt.plot(pose_xyz[:, 0],  pose_xyz[:, 1], label='Prediction', c='lime', linestyle='-')

        plt.legend()

        plt.xlabel('x (m)' )
        plt.ylabel('y (m)' )
        plt.grid(linestyle="--")
        png_title = "sequence_{}".format(output)
        fig_pdf = "plots/"+png_title + ".pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def plot_trajectory(poses_gt, poses_preds, output):
        """Plot trajectory for both GT and prediction
        """
        
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        pose_xyz = []
        for pose in poses_preds:
                pose_xyz.append([pose[0, 3], pose[1, 3], pose[2, 3]])
        pose_xyz = np.array(pose_xyz)
        plt.plot(-pose_xyz[:, 1],  -pose_xyz[:, 2], label='Prediction', c='lime', linestyle='-')

        traj_len = len(pose_xyz)
        pose_xyz = []
        for i,pose in enumerate(poses_gt):
            pose_xyz.append([pose[0, 3], pose[1, 3], pose[2, 3]])
            if i>traj_len: break
        pose_xyz = np.array(pose_xyz)
        plt.plot(pose_xyz[:, 0],  pose_xyz[:, 2], label='Ground Truth', c='k', linestyle='--')


        plt.legend()

        plt.xlabel('x (m)' )
        plt.ylabel('y (m)' )
        plt.grid(linestyle="--")
        png_title = "sequence_{}".format(output)
        fig_pdf = "plots/"+png_title + ".pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def load_poses_from_txt(file_name):
    """
    Poses are expressed by a 3x4 matrix composed of rotation matrix R and translation vector t.
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = []
    for line in s:
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i!=""]
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col]
        poses.append(P)
    return poses

def load_poses_from_txt_gt_drone(file_name):
    """
    poses for this dataset are expressed as 9 values, 
    the values at index 1,2,3 indicate the position x,y,z.
    We are interested in the position x,y
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    X, Y = [], []
    with open(file_name, 'r') as f:
        file = f.read()
        lines = file.split('\n')[:-1]
    pose = [[float(line.split(' ')[1]), float(line.split(' ')[2])] for line in lines]
    
    return pose


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
