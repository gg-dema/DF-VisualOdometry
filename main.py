from flow_net.flow_net_ui import Flow_net_ui 
from depth_net.depth_net import Depth_net
import numpy as np
import PIL.Image
from torchvision import transforms
from sklearn import linear_model
from tracker.tracker_v4 import TrackerInterface
import pickle
import torch
from scale_recovery import simple_scale_recovery
import os
from utils import *



def main():
    default_camera_calib = [
        [718.856, 0.00000, 607.1928],
        [0.00000, 718.856, 185.2157],
        [0.00000, 0.00000, 1.000000],
    ]

    seq = "02"

    ground_truth = load_poses_from_txt("data_odometry_poses/dataset/poses/"+seq+".txt")
    gt_0 = ground_truth[0]
    cam_prop, config_dict = cam_stuff(seq)
    tracker_ui = TrackerInterface(cam_prop, config_dict)



    flow_ui = Flow_net_ui()
    depNetwork = Depth_net()

    folder_path = 'data_odometry_color/dataset/sequences/'+seq+'/image_2'
    i = -1
    
    # Iterate through the files in the folder
    save = True
    if save:
        file = open("preds/preds_"+seq+"_trackerv4_pnp.txt", 'ab')
    for filename in os.listdir(folder_path):    
        
        file_path = os.path.join(folder_path, filename)
        if i == -1:
            abs_pred = np.eye(4)
            path_2 = file_path
            i += 1
            continue
            
        i += 1
        curr_gt = ground_truth[i-1]#because gt start from 0, but gt[0] corresponds to the transition from frame 0 to frame 1, with is computed when i = 1
        
        path_1 = path_2
        path_2 = file_path

        #is forward really needed here?
        cols, rows, matched_cols, matched_rows = flow_ui.get_matches(path_1, path_2)

        kp1 = np.array((rows, cols)).T
        kp2 = np.array((matched_rows, matched_cols)).T
        depth = depNetwork.predict(path_1)
        row_indices_tensor = torch.tensor(rows)
        col_indices_tensor = torch.tensor(cols)

        # Ensure depth map and index tensors are of the same data type
        depth_map_tensor = depth.to(torch.float32)
        row_indices_tensor = row_indices_tensor.to(torch.int64)

        # Compute flattened indices based on row and column indices
        flattened_indices = row_indices_tensor * depth_map_tensor.shape[1] + col_indices_tensor

        # Collect depth values for each point using tensor indexing
        z = depth_map_tensor.view(-1).gather(0, flattened_indices)
        z = z.detach().numpy()    

        pose = tracker_ui.get_pose_from_2d(kp1, kp2)
        if pose is None:
             print("E-tracker failure, using Pnp")
             pose = tracker_ui.get_pose_from_3d(kp1, kp2, z)
        R = pose['R']
        t = pose['t'].flatten().T
        s=1
        #s = simple_scale_recovery(kp1.T, kp2.T, R, t, torch.FloatTensor(cam_prop.intrinsics_matrix), z.detach().numpy())
        #st = s*t
        """
        curr_pred = np.zeros(size=(4,4))
        curr_pred[:3,:3] = R
        curr_pred[:3, 3] = t #st
        curr_pred[3,3] = 1"""
        #I'm gonna work with absolute pose, as thats what they do in the paper
        curr_pred = np.eye(4) #init current absolute pose
        curr_pred[:3,3] = abs_pred[:3,:3] @ t + abs_pred[:3,3] #compute current absolute translation
        curr_pred[:3,:3] = abs_pred[:3,:3] @ R #compute current absolute rotation
        abs_pred = curr_pred #update the absolute pose predicted so far

        if save:
                
                line = ' '.join(str(value) for value in curr_pred[:3,:].flatten())
                line += '\n'
                file.write(line.encode())

        
        if i == 1:
            pred_0 = curr_pred
        r_error, t_error = compute_error(gt_0 = np.eye(4), pred_0 = np.eye(4), curr_gt = curr_gt, curr_pred = curr_pred)
        print(i, r_error, t_error, s)

    preds = load_poses_from_txt("preds/preds_"+seq+"_trackerv4_pnp.txt")
    pred_0 = preds[0]
    curr_pred = np.eye(4)
    gt_0=ground_truth[0]
    previous_gt = np.eye(4) 
    abs_pred = np.eye(4)
    for i in range(len(preds)-1):
        break
        rel_gt = (previous_gt.T) @ ground_truth[i]
        #preds[i][:3, 3] = rel_gt[:3, 3]
        #curr_pred = curr_pred@preds[i+1] #ABSOLUTE
        #print(i, rotationMatrixToEulerAngles(curr_pred[:3,:3]), rotationMatrixToEulerAngles(rel_gt[:3,:3]))

        #curr_pred = preds[i+1] #RELATIVE
        #print(i, rotationMatrixToEulerAngles(curr_pred[:3,:3]), rotationMatrixToEulerAngles(rel_gt[:3,:3]))
        #random = np.random.randn(3,3)
        #print(curr_pred[:3,:3], random)
        #curr_pred[:3,:3] = np.zeros(shape=(3,3))

        #Trying a absolute pose method based on their git, OK IT'S THE SAME OPERATION WE DID, JUST WRITTEN DIFFERENTLY
        new_abs_pred = np.eye(4)
        new_abs_pred[:3,3] = abs_pred[:3,:3] @ preds[i][:3,3] + abs_pred[:3,3]
        new_abs_pred[:3,:3] = abs_pred[:3,:3] @ preds[i][:3,:3]
        abs_pred = new_abs_pred
        preds[i] = abs_pred
        #preds[i]=curr_pred
        r_error, t_error = compute_error(gt_0 = gt_0, pred_0 = pred_0, curr_gt = ground_truth[i], curr_pred = abs_pred)
        print(i, r_error, t_error, abs_pred[:3,3], ground_truth[i][:3,3])
        print("_________________________________")
        #previous_gt = ground_truth[i]
    plot_trajectory(ground_truth, preds, 22)

if __name__ == '__main__':
    main()



