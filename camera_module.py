import numpy as np


class CameraProperty:

    def __init__(self, intrinsics_matrix: np.array, ):

        if not type(intrinsics_matrix) == np.array:
            intrinsics_matrix = np.array(intrinsics_matrix)
            print('convert camera matrix to numpy, pls next time check the type')
        assert (intrinsics_matrix.shape[0], intrinsics_matrix.shape[1]) == (3, 3)

        self.intrinsics_matrix = intrinsics_matrix
        self.fx = intrinsics_matrix[0][0]
        self.fy = intrinsics_matrix[1][1]
        self.px_0 = intrinsics_matrix[0][2]
        self.py_0 = intrinsics_matrix[1][2]

