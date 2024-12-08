from abc import ABC, abstractmethod

import cv2
import numpy as np


class PoseEstimator(ABC):
    def __init__(self, points3d: np.ndarray, points2d: np.ndarray, **kwargs):
        self._points3d = points3d
        self._points2d = points2d
        self.R = None
        self.t = None
        self.inliers = None

        self._estimate()

    @abstractmethod
    def _estimate(self):
        pass


class PnPRansac(PoseEstimator):
    def __init__(self, points3d: np.ndarray, points2d: np.ndarray, **kwargs):
        self._K = kwargs.get('K')
        self._max_iter = kwargs.get('ransac_max_it', 100)
        self._dist_coeffs = kwargs.get('dist_coeffs', None)

        super().__init__(points3d, points2d, **kwargs)

    def _estimate(self):
        if self._points3d.shape[0] < 4 or self._points2d.shape[0] < 4:
            return

        # SolvePnP with RANSAC to handle outliers
        success, rvec, tvec, self.inliers = cv2.solvePnPRansac(
            objectPoints=self._points3d,  # 3D points in world space
            imagePoints=self._points2d,  # 2D points in image space
            cameraMatrix=self._K,  # Intrinsic camera matrix
            distCoeffs=self._dist_coeffs,  # Distortion coefficients (None if undistorted)
            reprojectionError=8.0,
            iterationsCount=self._max_iter,
            flags=cv2.SOLVEPNP_ITERATIVE  # Standard iterative approach
        )

        if not success:
            return

        # Back to rotation matrix
        self.R, _ = cv2.Rodrigues(rvec)
        self.t = tvec
