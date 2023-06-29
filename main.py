from flow_net.flow_net_ui import Flow_net_ui 
from flow_net.flow_net_ui_older import Flow_net_ui_older 
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
import cv2



def main():
    """
    This is now useless, it was just to verify that the intrinsic matrix was correct
    
    P=np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 4.538225000000e+01],
        [0.000000000000e+00 ,7.188560000000e+02, 1.852157000000e+02, -1.130887000000e-01 ],
        [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 3.779761000000e-03],])
    K, E = decompose_projection_matrix(P)
    """
    
    seq = "02" #sequence with a lot of curves, so useful for evaluation
    ground_truth = load_poses_from_txt("data_odometry_poses/dataset/poses/"+seq+".txt")
    gt_0 = ground_truth[0]
    cam_prop, config_dict = cam_stuff(seq)
    tracker_ui = TrackerInterface(cam_prop, config_dict)


    #init flow and depth net
    flow_ui = Flow_net_ui()
    depNetwork = Depth_net()

    #folder of sequence
    folder_path = 'data_odometry_color/dataset/sequences/'+seq+'/image_2'

    #where to save outputs
    save_path="test_17_6_normalized"
    save = True #wheter to save the outputs, maybe put it as a command line arg? 
    preds = {}
    if save:
        file = open("preds/preds_"+save_path+".txt", 'ab')
    i = -1
    #iterate through all the images in the sequence path, so trhough the consecutive frames
    for filename in os.listdir(folder_path):    
        #If you want to skip making the predictions, and just plot the trajectory from an already saved trajectory, uncomment break
        #break

        file_path = os.path.join(folder_path, filename)
        #to make a prediction we need consecutive frames, so we skip the first one, and start by comparing the first frame with the second
        if i == -1:
            #we initialize the absolute pose predicted with the first element of the ground truth. I tought this would solve the problem of having
            #our prediction go the wrong way from the start, but it didn't.
            abs_pred = ground_truth[0]
            path_2 = file_path
            i += 1
            continue
        
            
        i += 1
        curr_gt = ground_truth[i-1]#because gt start from 0, but gt[0] corresponds to the transition from frame 0 to frame 1, with is computed when i = 1
        
        
        path_1 = path_2
        path_2 = file_path

        #obtain matches from flow net
        cols, rows, matched_cols, matched_rows = flow_ui.get_matches(path_1, path_2, mode = "local_top_k", draw = False)
        row_indices_tensor = torch.tensor(rows)
        col_indices_tensor = torch.tensor(cols)
        """Check using random keypoints, to confirm that using random points leads to random trajectory.
        
        #rows = np.random.randint(0, 192, size=len(rows))
        #cols = np.random.randint(0, 640, size=len(cols))
        #matched_rows = np.random.randint(0, 192, size=len(rows))
        #matched_cols = np.random.randint(0, 640, size=len(cols))
        """

        """Normalized coordinates - not working
        We need to fix this
        centroid_x_1 = np.mean(cols)
        centroid_y_1 = np.mean(rows)
        centroid_x_2 = np.mean(matched_cols)
        centroid_y_2 = np.mean(matched_rows)

        
        translated_x_1 = cols - centroid_x_1
        translated_y_1 = rows - centroid_y_1
        translated_x_2 = matched_cols - centroid_x_2
        translated_y_2 = matched_rows - centroid_y_2

        
        avg_distance_x_1 = np.mean(np.linalg.norm(translated_x_1))
        avg_distance_y_1 = np.mean(np.linalg.norm(translated_y_1))
        avg_distance_x_2 = np.mean(np.linalg.norm(translated_x_2))
        avg_distance_y_2 = np.mean(np.linalg.norm(translated_y_2))

        
        cols = translated_x_1 / avg_distance_x_1
        rows = translated_y_1 / avg_distance_y_1
        matched_cols = translated_x_2 / avg_distance_x_2
        matched_rows = translated_y_2 / avg_distance_y_2"""


        kp1 = np.array((rows, cols)).T.astype(np.float64)
        kp2 = np.array((matched_rows, matched_cols)).T.astype(np.float64)
        
        #TODO this in numpy 
        #Obtain the depth values of each point [row, col] in frame 1
        depth = depNetwork.predict(path_1)
        depth_map_tensor = depth.to(torch.float32)
        row_indices_tensor = row_indices_tensor.to(torch.int64)

        flattened_indices = row_indices_tensor * depth_map_tensor.shape[1] + col_indices_tensor

        z = depth_map_tensor.view(-1).gather(0, flattened_indices)
        z = z.detach().numpy()    

        pose = tracker_ui.get_pose_from_2d(kp1, kp2)
        if pose is None:
            print("E-tracker failure, using Pnp")
            pose = tracker_ui.get_pose_from_3d(kp1, kp2, z)
        R = pose['R'].T
        t = pose['t'].flatten().T
        s=1
        
        #scale recovery, at the moment it is commented out, and s is set to 1 as we still have things to fix beforehand
        #s = simple_scale_recovery(kp1.T, kp2.T, R, t, cam_prop.intrinsics_matrix, z)
        #t = s*t
        curr_pred = np.eye(4) #init current absolute pose
        curr_pred[:3,3] = abs_pred[:3,:3] @ t + abs_pred[:3,3] #compute current absolute translation
        curr_pred[:3,:3] = abs_pred[:3,:3] @ R #compute current absolute rotation
        print(curr_pred)
        abs_pred = curr_pred #update the absolute pose predicted so far

        #save current absolute pose
        if save:
                line = ' '.join(str(value) for value in curr_pred[:3,:].flatten())
                line += '\n'
                file.write(line.encode())
        preds[i-1]=curr_pred


        r_error, t_error = compute_error(gt_0 = np.eye(4), pred_0 = np.eye(4), curr_gt = curr_gt, curr_pred = curr_pred)
        print("_______________frame_numer_"+str(i)+"_____________________")
        print("rotation around euler axis for relative rotation: ", rotationMatrixToEulerAngles(R), "relative translation:", t)#printing rotation around y
        print("rotation around euler axis for ground truth: ", rotationMatrixToEulerAngles(curr_gt[:3,:3] ), "and predicted:", rotationMatrixToEulerAngles(curr_pred[:3,:3] ))#printing rotation around y
        print("translation vector gt: ",curr_gt[:3,3],"predicted:", curr_pred[:3,3])#printing the translation vector at each iteration
        #if you want to stop after 150 frames change to True
        if i == 150 and False:
             break

    #PLOTTING TRAJECTORY
    preds = load_poses_from_txt("preds/preds_"+save_path+".txt")
    pred_0 = preds[0]
    curr_pred = np.eye(4)
    gt_0=ground_truth[0]
    previous_gt = np.eye(4) 
    abs_pred = np.eye(4)
    for i in range(len(preds)-1):
        #prints information about prediction, ground truth ecc for each step, this isn't needed to plot the trajecotry, but just to get more info
        #uncomment break to skip
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
    print(len(preds))

    #actual plot of trajectories
    plot_trajectory(ground_truth, preds, save_path)

if __name__ == '__main__':
    main()



