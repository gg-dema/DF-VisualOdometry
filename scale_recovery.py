from flow_net.flow_utils import Network, estimate
from depth_net.depth_net import Depth_net
import numpy as np
import PIL.Image
from torchvision import transforms
from sklearn import linear_model

import torch
import cv2 as cv

#FLOW_THRESHOLD = 10

def estimate_scaling_factor(d, d_prime):
    ransac = linear_model.RANSACRegressor(
                        estimator=linear_model.LinearRegression(
                            fit_intercept=False),
                        min_samples=15,
                        max_trials=100,
                        stop_probability=0.99,
                        residual_threshold=0.1,
                        )
    
    d_mask = d>0
    d_prime_mask = d_prime>0
    count = 0
    for i in range(0):#(len(d_prime_mask)):
        if d_prime_mask[i] == True:
            print(i, d_prime_mask[i], d_prime[i])
            count+=1
    print(count)
    mask = d_mask*d_prime_mask
    d_non_zero = d*mask
    d_prime_non_zero = d_prime*mask
    ransac.fit(
        d_prime_non_zero.reshape(-1, 1),
        d_non_zero.reshape(-1, 1),
                )
    scale = ransac.estimator_.coef_[0, 0]
    return scale

def triangulate(kp1, kp2, R, t, K):
    eye = np.eye(3)
    zeros = np.zeros(t.shape).reshape(3, 1)

    P1_proj = np.matmul(K, np.concatenate((eye, zeros), axis=1))
    P2_proj = np.matmul(K, np.concatenate((R, t.reshape(3, 1)), axis=-1))
    print(kp1.shape)
    print(P1_proj)

    kp1_norm = kp1.copy()
    kp2_norm = kp2.copy()
    kp1_norm[:, 0] = \
        (kp1[:, 0] - K[2,0]) /  K[0,0]
    kp1_norm[:, 1] = \
        (kp1[:, 1] - K[2,1]) / K[1,1]
    kp2_norm[:, 0] = \
        (kp2[:, 0] - K[2,0]) / K[0,0]
    kp2_norm[:, 1] = \
        (kp2[:, 1] - K[2,1]) / K[1,1]
    triangulated_points = cv.triangulatePoints(P1_proj[:3], P2_proj[:3], kp1_norm.astype(np.float64), kp2_norm.astype(np.float64))
    triangulated_points = triangulated_points.astype(np.float64)
    print("triang_done")
    triangulated_points /= triangulated_points[3]
    X2 = P2_proj[:3] @ triangulated_points
    
    #points = cv.convertPointsFromHomogeneous(triangulated_points)
    print("DONE")
    #print(X2[2])
    #points_euclidean = triangulate_points(P1_proj.detach().numpy(), P2_proj.detach().numpy(), points, matched_points)
    return X2[2]

def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return (X / X[3])[2]


def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    X = []
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    for i in range(len(x1[0])):
        p1 = np.array([x1[0,i], x1[1,i], 1])
        p2 = np.array([x2[0,i], x2[1,i], 1])
        print(p1, p2)
        X.append(triangulate_nviews([P1, P2],[p1, p2]))
    print(X)
    return np.array(X)


def simple_scale_recovery(kp1, kp2, R, t, K, depth):
    
    d_prime = triangulate(kp1, kp2, R, t, K)
    return estimate_scaling_factor(depth, d_prime)
"""

K = torch.randn(size=(3, 3), requires_grad = False)
R = torch.randn(size=(3, 3), requires_grad = False)
t = torch.randn(size=(3, 1), requires_grad = False)
#print(depth[0,0])
netNetwork = Network().to("cuda")
netNetwork.eval()

depNetwork = Depth_net()
#000000_10
tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open("000000_10.png"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open("000000_11.png"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
with torch.no_grad():
    forward = estimate(tenOne, tenTwo, netNetwork)

depth = depNetwork.predict("000000_10.png")

import pickle
with open('matches.pkl', 'rb') as f:
    cols, rows, matched_cols, matched_rows = pickle.load(f)
kp1 = np.array((rows, cols))
kp2 = np.array((matched_rows, matched_cols))

row_indices_tensor = torch.tensor(rows)
col_indices_tensor = torch.tensor(cols)

# Ensure depth map and index tensors are of the same data type
depth_map_tensor = depth.to(torch.float32)
row_indices_tensor = row_indices_tensor.to(torch.int64)

# Compute flattened indices based on row and column indices
flattened_indices = row_indices_tensor * depth_map_tensor.shape[1] + col_indices_tensor

# Collect depth values for each point using tensor indexing
depth_values = depth_map_tensor.view(-1).gather(0, flattened_indices)

s=simple_scale_recovery(kp1, kp2, R, t, K, depth_values.detach().numpy())
print("AHHHHHHHH", s)




def scale_recovery(R, t, K, forward, depth, s=0, iters=10):
    for i in range(iters):
        T = torch.cat((R, s*t), dim=-1)
        rigid_flow=torch.zeros(size=forward.size(), requires_grad=False)
        points = {"original": [],
                  "matches": []}
        for x in range(len(forward[1])):
            for y in range(len(forward[0])):
                p1 = torch.FloatTensor([x,y,0])
                p1.requires_grad = False
                #rigid_flow = R*p1 + t - x #t/depth[x]? according to slides it seems so
                K_inv = torch.inverse(K)

                r = K @ R @ K_inv @ p1 + K @ (t/depth[x,y])[:,0] - p1
                
                rigid_flow[0,x,y] = r[0]
                rigid_flow[1,x,y] = r[1]
                flow_inconsistency = (forward[0, x, y] - rigid_flow[0, x, y])**2 \
                                        + (forward[1, x, y] - rigid_flow[1, x, y])**2
        
                if flow_inconsistency>FLOW_THRESHOLD:#TO FIX
                    points["original"].append((p1[:2]).detach().numpy())
                    points["matches"].append(((p1+r)[:2]).detach().numpy())
        original = np.array(points["original"]).reshape(-1, 1, 2)
        matches = np.array(points["matches"]).reshape(-1, 1, 2)
        #R, t = estimate_pose(points["original"], points["matches"])
        with torch.no_grad():
            d_prime = triangulate(original, matches, R, t, K)#TODO com cazz se fa
        s = estimate_scaling_factor(depth.flatten(), d_prime) #??? paper just says s is estimated by comparing depth and d_prime
        print(s)"""