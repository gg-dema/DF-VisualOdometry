import os
import torch
import numpy as np
from flow_net.flow_net_ui import Flow_net_ui 
from depth_net.depth_net import Depth_net
from tracker.tracker_v4 import TrackerInterface
from utils import *


class VisualOdometry:
    
    def __init__(self, args):
        seq = "02" #sequence with a lot of curves, so useful for evaluation
        cam_prop, config_dict = self.cam_stuff(args.dataset)
        self.tracker_ui = TrackerInterface(cam_prop, config_dict)
        self.flow_ui = Flow_net_ui()
        self.depNetwork = Depth_net()
        self.folder_path = args.input
        self.dataset = args.dataset
        self.save_path= args.output

    def compute_trajectory(self):
        file = open("preds/preds_"+self.save_path+".txt", 'ab')
        i = -1
        #iterate through all the images in the sequence path, so trhough the consecutive frames
        for filename in os.listdir(self.folder_path):    
            file_path = os.path.join(self.folder_path, filename)
            #to make a prediction we need consecutive frames, so we skip the first one, and start by comparing the first frame with the second
            if i == -1:
                #we initialize the absolute pose predicted with the first element of the ground truth. I tought this would solve the problem of having
                #our prediction go the wrong way from the start, but it didn't.
                abs_pred = np.eye(4)
                path_2 = file_path
                i += 1
                continue
                
            i += 1
            
            path_1 = path_2
            path_2 = file_path
            print(path_1, path_2)
            #obtain matches from flow net
            cols, rows, matched_cols, matched_rows = self.flow_ui.get_matches(path_1, path_2, mode = "local_top_k", draw = False, dataset=self.dataset)

            kp1 = np.array((rows, cols)).T.astype(np.float64)
            kp2 = np.array((matched_rows, matched_cols)).T.astype(np.float64)
            
            #Obtain the depth values of each point [row, col] in frame 1
            depth = self.depNetwork.predict(path_1)
            depth_map_tensor = depth.to(torch.float32)

            flattened_indices = torch.tensor(rows) * depth_map_tensor.shape[1] + torch.tensor(cols)

            z = depth_map_tensor.view(-1).gather(0, flattened_indices)
            z = z.detach().numpy()    

            pose = self.tracker_ui.get_pose_from_2d(kp1, kp2)

            if pose is None:
                print("E-tracker failure, using Pnp")
                pose = self.tracker_ui.get_pose_from_3d(kp1, kp2, z)
            
            R = pose['R'].T
            t = pose['t'].flatten().T

            curr_pred = np.eye(4) #init current absolute pose
            curr_pred[:3,3] = abs_pred[:3,:3] @ t + abs_pred[:3,3] #compute current absolute translation
            curr_pred[:3,:3] = abs_pred[:3,:3] @ R #compute current absolute rotation
            abs_pred = curr_pred #update the absolute pose predicted so far

            #save current absolute pose
            
            line = ' '.join(str(value) for value in curr_pred[:3,:].flatten())
            line += '\n'
            file.write(line.encode())

    def cam_stuff(self, cam_type):
        #Intrinsic matrix depends on the dataset(and correspoinding camera)
        if cam_type == 'drone': 
            intrinsic_matrix = np.array([[446.38011635796755, 0, 306.68683920642184],[0, 445.88719476806966, 246.3972559515397],[0, 0, 1]])
        else:
            intrinsic_matrix=np.array([[7.188560000000e+02, 0, 6.071928000000e+02], [0 ,7.188560000000e+02, 1.852157000000e+02],[0, 0, 1]])

        cam_prop = CameraProperty(np.array(intrinsic_matrix))
        config_dict = {'robust_tracker': False,
                    'numb_robust_iter': 10}
        return cam_prop, config_dict