import numpy as np
import math
from camera_module import CameraProperty
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

def plot_trajectory_drone(poses_gt, poses_preds, output):
        """Plot trajectory for both GT and prediction
        We make some adjustments to the axis, to account for position of the camera w.r.t World frame
        """
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        pose_xy = poses_gt
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
        We make some adjustments to the axis, to account for position of the camera w.r.t World frame
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


def compute_translation_error(gt, preds):
    #Compute the translation error within the last position of the ground truth and our prediction
    last_pose_gt = np.array(gt[len(preds)])
    last_pose_pred = np.array(preds[-1])

    total = (last_pose_gt[0,3] - (-last_pose_pred[1,3]))**2 + (last_pose_gt[2,3] - (-last_pose_pred[2,3]))**2
    
    #Comput the relative translation error within consecutive frames
    old_pose_gt = np.array(gt[0])
    old_pose_pred = np.array(preds[0])
    total_rel = 0

    for i in range(len(preds)):
        curr_pose_gt = np.array(gt[i])
        curr_pose_pred = np.array(preds[i])

        relative = (  (curr_pose_gt[0,3] - old_pose_gt[0,3]) - (-curr_pose_pred[1,3] + old_pose_pred[1,3])  )**2 \
                    + (  (curr_pose_gt[2,3] - old_pose_gt[2,3])  - (-curr_pose_pred[2,3] + old_pose_pred[2,3])    )**2

        total_rel += relative
        old_pose_gt = curr_pose_gt
        old_pose_pred = curr_pose_pred
    return total, total_rel/len(preds)

def compute_translation_error_drone(gt, preds):
    #Compute the translation error within the last position of the ground truth and our prediction
    
    last_pose_gt = np.array(gt[-1])
    last_pose_pred = np.array(preds[-1])
    print(last_pose_gt[0], last_pose_pred[0,3], last_pose_gt[1], last_pose_pred[1,3])
    total = (last_pose_gt[0] - (-last_pose_pred[0,3]))**2 + (-last_pose_gt[1] - last_pose_pred[1,3])**2

    #The ground truth provided for the drone daatset doesn't allow us to compute the relative translation error sadly
    
    return total