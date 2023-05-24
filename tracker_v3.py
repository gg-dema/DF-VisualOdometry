from typing import Optional, Dict
from camera_module import CameraProperty
import numpy as np
import cv2


# ------------------------------------------------
# GRIC SCORE CALC PART --> MODEL SELECTION
# bunch of function, cannot access to this outisde
# ------------------------------------------------

def calc_GRIC(res, sigma, n, model: str):
    """Calculate GRIC

    Args:
        res (array, [N]): residual
        sigma (float): assumed variance of the error
        n (int): number of residuals
        model (str): model type
            - FMat
            - EMat
            - HMat
    """
    R = 4
    sigma_sq1 = 1. / sigma ** 2

    # model = one of the possible option, just wrap a big if
    K = {"FMat": 7, "EMat": 5, "HMat": 8, }[model]
    D = {"FMat": 3, "EMat": 3, "HMat": 2, }[model]

    lam3RD = 2.0 * (R - D)

    sum_ = 0
    for i in range(n):
        tmp = res[i] * sigma_sq1
        if tmp <= lam3RD:
            sum_ += tmp
        else:
            sum_ += lam3RD

    sum_ += n * D * np.log(R) + K * np.log(R * n)

    return sum_


def calc_residual_H(H_in, kp1, kp2):
    """
    Compute homography matrix residual

    Args:
        H (array, [3x3]): homography matrix (Transformation from view-1 to view-2)
        kp1 (array, [Nx2]): keypoint 1
        kp2 (array, [Nx2]): keypoint 2

    Returns:
        res (array, [N]): residual
    """
    n = kp1.shape[0]
    H = H_in.flatten()

    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1, 0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1, 0))

    G0 = np.zeros((3, n))
    G1 = np.zeros((3, n))

    G0[0] = H[0] - m1[0] * H[6]
    G0[1] = H[1] - m1[0] * H[7]
    G0[2] = -m0[0] * H[6] - m0[1] * H[7] - H[8]

    G1[0] = H[3] - m1[1] * H[6]
    G1[1] = H[4] - m1[1] * H[7]
    G1[2] = -m0[0] * H[6] - m0[1] * H[7] - H[8]

    magG0 = np.sqrt(G0[0] * G0[0] + G0[1] * G0[1] + G0[2] * G0[2])
    magG1 = np.sqrt(G1[0] * G1[0] + G1[1] * G1[1] + G1[2] * G1[2])
    magG0G1 = G0[0] * G1[0] + G0[1] * G1[1]

    alpha = np.arccos(magG0G1 / (magG0 * magG1))

    alg = np.zeros((2, n))
    alg[0] = m0[0] * H[0] + m0[1] * H[1] + H[2] - \
             m1[0] * (m0[0] * H[6] + m0[1] * H[7] + H[8])

    alg[1] = m0[0] * H[3] + m0[1] * H[4] + H[5] - \
             m1[1] * (m0[0] * H[6] + m0[1] * H[7] + H[8])

    D1 = alg[0] / magG0
    D2 = alg[1] / magG1

    res = (D1 * D1 + D2 * D2 - 2.0 * D1 * D2 * np.cos(alpha)) / np.sin(alpha)

    return res


def calc_residual_F(F, kp1, kp2):
    """
    Compute fundamental matrix residual

    Args:
        F (array, [3x3]): Fundamental matrix (from view-1 to view-2)
        kp1 (array, [Nx2]): keypoint 1
        kp2 (array, [Nx2]): keypoint 2

    Returns:
        res (array, [N]): residual
    """
    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1, 0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1, 0))

    Fm0 = F @ m0  # 3xN
    Ftm1 = F.T @ m1  # 3xN

    m1Fm0 = (np.transpose(Fm0, (1, 0)) @ m1).diagonal()
    res = m1Fm0 ** 2 / (np.sum(Fm0[:2] ** 2, axis=0) + np.sum(Ftm1[:2] ** 2, axis=0))
    return res


# ------------------------------------------------
# TRACKER PART (epipolar, PnP, interface)
# ------------------------------------------------

class ETracker:

    def __init__(self, camera_prop: CameraProperty):
        self.camera_prop = camera_prop

    def simple_E_estimation(self, kp1, kp2) -> Optional[np.array]:
        """ Estimate E matrix without count inliers [ ranSac by openCv2 ] """
        E_matrix, _ = cv2.findEssentialMat(kp1, kp2)
        if not self.valid_estimation(E_matrix, kp1, kp2):
            return None
        return E_matrix

    def robust_E_estimation(self, kp1, kp2, iteration) -> Optional[np.array]:
        """ eval the Essential matrix keeping into account the inlier:"""
        E_matrix = ...
        # TODO : add iterative methods form paper: # http://www.hilandtom.com/tombotterill/Botterill-Mills-Green-2011-RefineE.pdf
        if self.valid_estimation(E_matrix, kp1, kp2):
            return E_matrix
        else:
            return None

    def valid_estimation(self, E_matrix, kp1, kp2) -> bool:

        res_E = calc_residual_F(E_matrix, kp1, kp2) # idk why we call resiudal F

        H, _ = cv2.findHomography(kp1, kp2)
        res_H = calc_residual_H(H, kp1, kp2)

        n_point = kp1.shape[0]
        good_approx = (
                calc_GRIC(res_H, sigma=0.8, n=n_point, model='HMat') >
                calc_GRIC(res_E, sigma=0.8, n=n_point, model='EMat')
        )
        return good_approx


    def pose_estimation(self, kp1, kp2, E_matrix) -> Dict:
        _, R, t, _ = cv2.recoverPose(E_matrix, kp1, kp2,
                                     focal=self.camera_prop.fx,
                                     pp=(self.camera_prop.px_0, self.camera_prop.py_0))
        return {'R': R, 't': t}


class PnpTracker:

    def __init__(self, camera_prop):
        self.camera_prop = camera_prop

    def pose_estimation(self, kp1, kp2) -> Dict:
        pass


class TrackerInterface:

    def __init__(self, camera_prop: CameraProperty, config_dict: dict):
        self.e_tracker = ETracker(camera_prop)
        self.pnp_tracker = PnpTracker(camera_prop)
        self.config_dict = config_dict

    def get_pose_from_2d(self,
                         kp1: np.array,
                         kp2: np.array,
                         ) -> Optional[Dict]:

        assert kp1.shape == kp2.shape

        # choose how we want estimate the E matrix
        if self.config_dict['robust_E_tracker']:
            valid_E = self.e_tracker.robust_E_estimation(kp1, kp2, iteration=self.config_dict['robust_iteration'])
        else:
            valid_E = self.e_tracker.simple_E_estimation(kp1, kp2)

        if not valid_E is None:
            return self.e_tracker.pose_estimation(kp1, kp2, E_matrix=valid_E)
        else:
            return None

    def get_pose_from_3d(self,
                         kp1: np.array,
                         kp2: np.array,
                         ) -> Dict:
        assert len(kp1.shape) == 3
        assert len(kp2.shape) == 2
        assert len(kp1) == len(kp2)

        return self.pnp_tracker.pose_estimation(kp1, kp2)


""" main just for testing """
if __name__ == '__main__':
    import pickle

    default_camera_calib = [
        [718.856, 0.00000, 607.1928],
        [0.00000, 718.856, 185.2157],
        [0.00000, 0.00000, 1.000000],
    ]

    cam_prop = CameraProperty(np.array(default_camera_calib))

    config_dict = {'robust_E_tracker': False, }
    tracker_ui = TrackerInterface(cam_prop, config_dict)

    # test 2d:
    with open('test/matches.pkl', 'rb') as f:
        cols, rows, matched_cols, matched_rows = pickle.load(f)

    kp1 = np.array((rows, cols)).T
    kp2 = np.array((matched_rows, matched_cols)).T

    pose = tracker_ui.get_pose_from_2d(kp1, kp2)
    print('R:', pose['R'])
    print('t', pose['t'])

    # TO INSERT:  giusto per visualizzare un possibile flusso di lavoro
    # pose_re_scaled = SCALE_POSE(pose, IDK_WTF_U_NEED)

    # test 3d
