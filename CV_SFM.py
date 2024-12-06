import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.utils import _show_correspondence_lines, _save_image


class OpenCVSFM:
    def __init__(self, K):
        self.K = K
        self.poses = []  # List of (R, t)
        self.points_3D = []  # List of triangulated 3D points

    def find_matches(self, img1, img2):
        """Detect and match features using OpenCV SIFT."""
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Extract x, y coordinates for original keypoints
        X1 = np.array([kp.pt[0] for kp in kp1])
        Y1 = np.array([kp.pt[1] for kp in kp1])
        X2 = np.array([kp.pt[0] for kp in kp2])
        Y2 = np.array([kp.pt[1] for kp in kp2])

        # Prepare matches array for visualization
        matches_array = np.array([[m.queryIdx, m.trainIdx] for m in matches])

        # Call _print_matches to visualize
        self._print_matches(
            image1=img1,
            image2=img2,
            X1=X1, Y1=Y1,
            X2=X2, Y2=Y2,
            matches=matches_array
        )

        return pts1, pts2

    def _print_matches(self, image1, image2, X1, X2, Y1, Y2, matches, output_path: str = 'output/vis_lines.jpg'):
        """Visualizes matches between features."""
        if len(matches) == 0:
            print('No matches to visualize')
            return

        c2 = _show_correspondence_lines(
            image1, image2,
            X1[matches[:, 0]], Y1[matches[:, 0]],
            X2[matches[:, 1]], Y2[matches[:, 1]]
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(c2)
        _save_image(output_path, c2)

    def compute_fundamental_matrix(self, pts1, pts2):
        """Compute the fundamental matrix using RANSAC."""
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        return F, mask

    def compute_essential_matrix(self, pts1, pts2):
        """Compute the essential matrix."""
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        return E, mask

    def recover_pose(self, E, pts1, pts2):
        """Recover the camera pose from the essential matrix."""
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, mask

    def compute_projection_matrix(self, R, t):
        """Compute the projection matrix from the camera pose."""
        RT = np.hstack((R, t.reshape(-1, 1)))  # Combine R and T to form [R | T]
        P = self.K @ RT
        return P

    @staticmethod
    def triangulate_point(x1, x2, P1, P2):
        # Create the system of equations for triangulation
        A = np.vstack([
            x1[0] * P1[2, :] - P1[0, :],
            x1[1] * P1[2, :] - P1[1, :],
            x2[0] * P2[2, :] - P2[0, :],
            x2[1] * P2[2, :] - P2[1, :]
        ])

        # Solve for the 3D point X using SVD
        U, S, Vt = np.linalg.svd(A)

        # Check if the smallest singular value is too close to zero
        if S[-1] < 1e-6:
            print("Warning: Poor triangulation quality; check input data.")
            return np.array([np.nan, np.nan, np.nan])

        X = Vt[-1]
        X /= X[3]  # Normalize to get homogeneous coordinates (X, Y, Z, 1)

        return X[:3]  # Return 3D point (X, Y, Z)

    def add_pose(self, R, t):
        """Add the current camera pose to the list."""
        self.poses.append((R, t))

    def add_points(self, points):
        """Add 3D points to the global list."""
        self.points_3D.extend(points)
