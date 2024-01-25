import numpy as np
from scipy.spatial import distance
import random

def no_cut(center, enzyme, radius, enzyme_name):
    return enzyme

def count_cut(center, enzyme, count, enzyme_name):
    """Cuts the binding site from the center expanding by one atom at a time till count"""

    center = np.array(center).reshape(1, -1)  # cast to 2D array
    distances = distance.cdist(center, enzyme[:, :3], "euclidean")
    enzyme = np.hstack((enzyme, distances.reshape(len(enzyme), 1)))
    enzyme_binding_site = enzyme[enzyme[:, 4].argsort()]

    return enzyme_binding_site[0:count,:4]

def random_cut(center, enzyme, count, enzyme_name):
    """Cuts the binding site from a "RANDOM CENTER" expanding by one atom at a time till count"""

    rand_ind =  random.randint(0,enzyme.shape[0]-1) 
    center = enzyme[rand_ind, :3]
    center = np.array(center).reshape(1, -1)  # cast to 2D array
    distances = distance.cdist(center, enzyme[:, :3], "euclidean")
    enzyme = np.hstack((enzyme, distances.reshape(len(enzyme), 1)))
    enzyme_binding_site = enzyme[enzyme[:, 4].argsort()]

    return enzyme_binding_site[0:count,:4]


def cubic_cut(center, enzyme, radius, enzyme_name):
    """tests the array coordinates to be in a square box 2r wide"""

    enzyme_binding_site = enzyme[
        np.logical_and(
            enzyme[:, 0] >= (center[0] - radius), enzyme[:, 0] <= (center[0] + radius)
        )
        & np.logical_and(
            enzyme[:, 1] >= (center[1] - radius), enzyme[:, 1] <= (center[1] + radius)
        )
        & np.logical_and(
            enzyme[:, 2] >= (center[2] - radius), enzyme[:, 2] <= (center[2] + radius)
        )
    ]

    return enzyme_binding_site


def radial_cut(center, enzyme, radius, enzyme_name):
    """test the array coordinates to be in a sphere r radius away from the center."""
    
    enzyme_binding_site = enzyme[
        ((enzyme[:, 0] - center[0]) ** 2)
        + ((enzyme[:, 1] - center[1]) ** 2)
        + ((enzyme[:, 2] - center[2]) ** 2)
        <= (radius ** 2)
    ]

    return enzyme_binding_site


def custom_cut(center, enzyme):
    """define your custom binding site cutter, just make sure to pass the center
    of the object as a tuple, so you can interchange it with the standard cut
    methods. Then define a function to return only the array rows wanted"""
    return enzyme
 
 
def point_transform(self, points, tx, ty, tz, rx=0, ry=0, rz=0):
        """Input:
          points: (N, 3)
          rx/y/z: in radians
        Output:
          points: (N, 3)
        """
        N = points.shape[0]
        points = np.hstack([points, np.ones((N, 1))])
        mat1 = np.eye(4)
        mat1[3, 0:3] = tx, ty, tz
        points = np.matmul(points, mat1)
        if rx != 0:
            mat = np.zeros((4, 4))
            mat[0, 0] = 1
            mat[3, 3] = 1
            mat[1, 1] = np.cos(rx)
            mat[1, 2] = -np.sin(rx)
            mat[2, 1] = np.sin(rx)
            mat[2, 2] = np.cos(rx)
            points = np.matmul(points, mat)
        if ry != 0:
            mat = np.zeros((4, 4))
            mat[1, 1] = 1
            mat[3, 3] = 1
            mat[0, 0] = np.cos(ry)
            mat[0, 2] = np.sin(ry)
            mat[2, 0] = -np.sin(ry)
            mat[2, 2] = np.cos(ry)
            points = np.matmul(points, mat)
        if rz != 0:
            mat = np.zeros((4, 4))
            mat[2, 2] = 1
            mat[3, 3] = 1
            mat[0, 0] = np.cos(rz)
            mat[0, 1] = -np.sin(rz)
            mat[1, 0] = np.sin(rz)
            mat[1, 1] = np.cos(rz)
            points = np.matmul(points, mat)
        return points[:, 0:3]

def augment_rot_data(self, coords):

        r_x = random.uniform(0, 2 * math.pi)  # rotation augmentations
        r_y = random.uniform(0, 2 * math.pi)
        r_z = random.uniform(0, 2 * math.pi)

        coords[:, :3] = self.point_transform(
            coords[:, :3], 0, 0, 0, rx=r_x, ry=r_y, rz=r_z
        )

        return coords