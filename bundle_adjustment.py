import numpy as np
from scipy.optimize import least_squares
import cv2

class BundleAdjustment:
    def __init__(self, num_cameras, num_points, camera_indices, point_indices, points_2d, camera_params, points_3d, K):
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.camera_indices = camera_indices
        self.point_indices = point_indices
        self.points_2d = points_2d
        self.camera_params = camera_params
        self.points_3d = points_3d
        self.K = K

    def sparse_bundle_adjustment(self):
        # Flatten the initial parameters
        initial_params = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))

        # Optimize
        result = least_squares(
            self.compute_residuals,
            initial_params,
            args=(self.num_cameras, self.num_points, self.camera_indices, self.point_indices, self.points_2d, self.K),
            verbose=2,
            jac='2-point',  # Sparse Jacobian approximation
            method='trf',  # Trust-region reflective
        )

        # Reshape results
        optimized_camera_params = result.x[:self.num_cameras * 6].reshape((self.num_cameras, 6))
        optimized_points_3d = result.x[self.num_cameras * 6:].reshape((self.num_points, 3))

        return optimized_camera_params, optimized_points_3d

    def project_point(self, point_3d, R, t, K):
        point_cam = R @ point_3d + t
        point_proj = K @ point_cam
        return point_proj[:2] / (point_proj[2] + 1e-9)

    def compute_residuals(self, params, num_cameras, num_points, camera_indices, point_indices, points_2d, K):
        # 6 params in total for each camera, 3 for Rotation, 3 for translation
        camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
        points_3d = params[num_cameras * 6:].reshape((num_points, 3))
        residuals = []

        for cam_idx, point_idx, observed_2d in zip(camera_indices, point_indices, points_2d):
            # Extract camera parameters
            R_vec = camera_params[cam_idx-1, :3]
            t = camera_params[cam_idx-1, 3:6]

            # Convert Rotation Rodrigues vector to rotation matrix
            R, _ = cv2.Rodrigues(R_vec)

            # Reproject the 3D point
            point_3d = points_3d[point_idx]
            #K = K_list[cam_idx-1]
            projected_2d = self.project_point(point_3d, R, t, K)

            # Compute residual (reprojection error)
            residuals.append(projected_2d - observed_2d)

        return np.concatenate(residuals)