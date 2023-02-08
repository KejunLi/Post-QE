#!/usr/bin/env python3
import numpy as np


class rotation_matrix(object):
    def __init__(self) -> None:
        print("Start rotation:")
        pass
    def rotation_matrix_euler(self, alpha: float, beta: float, gamma: float) -> None:
        """
        input rotation angles in degree (alpha in z, beta in y, gamma in x)
        """
        print("1st rotation about x-axis by {}°".format(gamma))
        print("2nd rotation about y-axis by {}°".format(beta))
        print("3rd rotation about z-axis by {}°".format(alpha))
        # rotation in x
        gamma = gamma / 180.0 * np.pi # convert to radian from degree

        rotation_x = np.matrix(
            [
                [1, 0, 0], 
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]
            ]
        )

        # rotation in y
        beta = beta / 180.0 * np.pi # convert to radian from degree

        rotation_y = np.matrix(
            [
                [np.cos(beta), 0, np.sin(beta)], 
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ]
        )

        # rotation in z
        alpha = alpha / 180.0 * np.pi # convert to radian from degree

        rotation_z = np.matrix(
            [
                [np.cos(alpha), -np.sin(alpha), 0], 
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]
            ]
        )

        rot = np.matmul(rotation_y, rotation_x)
        self.rot_mat_euler = np.matmul(rotation_z, rot)



    def rotation_matrix_rodrigues(self, vec1=[1,1,1], vec2=[0,0,1]) -> None:
        """
        =---------------------------------------------------------------------------
        +   rotation matrix that align vector 1 to vector 2
        +   https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d?rq=1
        =---------------------------------------------------------------------------
        """
        print("Rotation by Rodrigues Fomular from {} to {}".format(vec1, vec2))
        # normalize the vector (vec1) to rotate and the objective direction (vec2)
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = vec2/np.linalg.norm(vec2)
        # find the axis of rotation by the cross product of vec1 and vec2
        v = np.cross(vec1, vec2)
        c = np.dot(vec1, vec2) # cos(theta)
        s = np.linalg.norm(v) # sin(theta)
        skew_symm_mat_of_v = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ]
        )
        self.rot_mat_rodrigues = (
            np.identity(3) 
            + skew_symm_mat_of_v 
            + np.matmul(skew_symm_mat_of_v, skew_symm_mat_of_v) * (1-c)/s**2
        )

