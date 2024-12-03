import cv2
import numpy as np
from numpy.linalg import svd
from PIL import Image
from PIL.ExifTags import TAGS
from enum import Enum
from scipy.optimize import least_squares


class SensorType(Enum):
    MEDIUM_FORMAT = 1
    FULL_FRAME = 2
    CROP_FRAME = 3
    MICRO_FOUR_THIRD = 4
    ONE_INCH = 5
    SMARTPHONE = 6


class Util:
    @staticmethod
    def construct_K(imagePath, sensorType: SensorType):
        image = Image.open(imagePath)
        width, height = image.size
        exif_data = image._getexif()

        # Decode EXIF data and extract focal length
        focal_length = None
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "FocalLength":
                    # If focal length is usually a tuple (numerator, denominator)
                    if isinstance(value, tuple):
                        focal_length = value[0] / value[1]  # Convert to a single value
                    else:
                        focal_length = value
                    break
        else:
            print("No EXIF data. Cannot work with this image")
            raise Exception("No EXIF data. Cannot work with this image")

        if focal_length is None:
            print("No focal length data. Cannot work with this image")
            raise Exception("No focal length data. Cannot work with this image")

        sensor_height = 0.0
        sensor_width = 0.0

        if sensorType is SensorType.MEDIUM_FORMAT:
            sensor_width = 53.0
            sensor_height = 40.20
        elif sensorType is SensorType.FULL_FRAME:
            sensor_width = 35.0
            sensor_height = 24.0
        elif sensorType is SensorType.CROP_FRAME:
            sensor_width = 23.6
            sensor_height = 15.60
        elif sensorType is SensorType.MICRO_FOUR_THIRD:
            sensor_width = 17.0
            sensor_height = 13.0
        elif sensorType is SensorType.ONE_INCH:
            sensor_width = 12.80
            sensor_height = 9.60
        elif sensorType is SensorType.SMARTPHONE:
            sensor_width = 6.17
            sensor_height = 4.55

        fx = focal_length * width / sensor_width
        fy = focal_length * height / sensor_height
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        return K

    @staticmethod
    def compute_euclidean_distance(arr1, arr2):
        if arr2.shape[0] == 1:
            # arr2 is a single point
            return np.linalg.norm(arr1 - arr2, axis=1)
        # Full distance matrix
        return np.linalg.norm(arr1[:, np.newaxis] - arr2, axis=2)


class CameraPose:
    def __init__(self, pts1, pts2, K1, K2):
        self.pts1 = pts1
        self.pts2 = pts2
        self.K1 = K1
        self.K2 = K2

    def normalize_points(self, points):
        mean = np.mean(points[:, :2], axis=0)
        cu, cv = mean[0], mean[1]

        # Euclidean dist
        sqr_dist = np.sqrt((points[:, 0] - cu) ** 2 + (points[:, 1] - cv) ** 2)
        mean_sqr_dist = np.mean(sqr_dist)
        scale = np.sqrt(2) / mean_sqr_dist

        T = np.array([[scale, 0, -scale * cu],
                      [0, scale, -scale * cv],
                      [0, 0, 1]])

        points_normalized = points @ T.T

        return points_normalized, T

    def unnormalize_F(self, F_norm, T_a, T_b):
        return T_b.T @ F_norm @ T_a

    @staticmethod
    def calculate_num_ransac_iterations(prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
        num_samples = np.log(1 - prob_success) / np.log(1 - (ind_prob_correct ** sample_size))
        return int(num_samples)

    def ransac_camera_motion(self, R_base, T_base, threshold=1.0, max_iterations=1000):
        best_inliers1, best_inliers2 = [], []
        best_r, best_t = None, None

        if len(self.pts1) < 8:
            return None, None, None, None

        for _ in range(max_iterations):
            # Randomly sample 8 correspondences
            indices = np.random.choice(len(self.pts1), 8, replace=False)
            sample_points1 = self.pts1[indices]
            sample_points2 = self.pts2[indices]

            # Compute the fundamental matrix using the 8-point algorithm
            F = self._compute_fundamental_matrix(sample_points1, sample_points2)

            # Decompose the fundamental matrix to obtain R and T
            # In our case, we assume that the camera intrinsic is the same for both image
            E = self.K2.T @ F @ self.K1

            # SVD on the essential matrix
            # U and Vt are rotation and reflection, S is scaling
            U, S, Vt = svd(E)

            # Rotation matrix is U and Vt
            W = np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
            R1 = np.dot(U, np.dot(W, Vt))
            R2 = np.dot(U, np.dot(W.T, Vt))

            # Translation vector is the last column of U
            T = U[:, 2]

            # Compute the projection error for all four combinations of R and T
            for R_candidate, T_candidate in [(R1, T), (R1, -T), (R2, T), (R2, -T)]:
                if not self._check_valid_pose(R_base, T_base, R_candidate, T_candidate):
                    continue  # Skip invalid pose

                inliers1, inliers2 = [], []

                for i in range(len(self.pts1)):
                    x1 = self.pts1[i]
                    x2 = self.pts2[i]

                    # Epipolar line of point a: l_1 = F.P_2 due to epipolar constraint
                    la = F @ np.array([x1[0], x1[1], 1])

                    # Dist is |la . P2| / norm(la)
                    distance = np.abs(la @ np.array([x2[0], x2[1], 1])) / np.sqrt(la[0] ** 2 + la[1] ** 2)
                    if distance < threshold:
                        inliers1.append(x1)
                        inliers2.append(x2)

                # If this rotation and translation give more inliers, select it as the best
                if len(inliers1) > len(best_inliers1):
                    best_inliers1 = inliers1
                    best_inliers2 = inliers2
                    best_r, best_t = R_candidate, T_candidate

        return best_r, best_t, np.array(best_inliers1), np.array(best_inliers2)

    def _compute_fundamental_matrix(self, p1, p2) -> np.ndarray:
        # Ensure points are in homogeneous coordinates
        n = p1.shape[0]
        pts1_hom, T1 = self.normalize_points(np.hstack([p1, np.ones((n, 1))]))
        pts2_hom, T2 = self.normalize_points(np.hstack([p2, np.ones((n, 1))]))

        # Build the matrix A, where each row corresponds to a correspondence
        # The derivation is as followed:
        # Fundamental equ: l_2 = F.P_1 due to epipolar constraint
        #                  l_2 . P_2 = P2.t.(F.P_1)
        #                  0 = P_2.t.(F.P_1) since P_2 lies on epipolar line l2, hence their dot prod is 0
        # Expand: [x_2 y_2 1] . [[f_11 f_12 f_13]  . [x_1  = 0
        #                        [f_21 f_22 f_23]     y_1
        #                        [f_31 f_32 f_33]]     1]
        # => [x_2 y_2 1] . [[f_11x_1 + f_12y_1 + f_13] = 0
        #                   [f_21x_1 + f_22y_1 + f_23]
        #                   [f_31x_1 + f_32y_1 + f_33]]
        # => f_11x_1x_2 + f_12y_1x_2 + f_13x_2 + f_21x_1y_2 + f_22y_1y_2 + f_23y_2 + f_31x_1 + f_32y_1 + f_33 = 0
        # => [x_1x_2 + y_1x_2 + x_2 + x_1y_2 + x_1y_2 + y_1y_2 + y_2 + x_1 + y_1 + 1] [f_11 = 0
        #                                                                              f_12
        #                                                                              f_13
        #                                                                              f_21
        #                                                                              f_23
        #                                                                              f_31
        #                                                                              f_32
        #                                                                              f_33]
        A = np.zeros((n, 9))
        for i in range(n):
            x1, y1 = pts1_hom[i, 0], pts1_hom[i, 1]
            x2, y2 = pts2_hom[i, 0], pts2_hom[i, 1]
            A[i] = [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]

        # Solve for the vector f (flattened fundamental matrix) using SVD
        _, _, VT = np.linalg.svd(A)
        F_vec = VT[-1, :]

        # Reshape into 3x3 matrix
        F = F_vec.reshape(3, 3)

        # Enforce the rank-2 constraint (set the smallest singular value to zero)
        U, D, Vt = np.linalg.svd(F)
        D[2] = 0  # Set the smallest singular value to 0
        F_rank2 = np.dot(U, np.dot(np.diag(D), Vt))

        F_rank2 = self.unnormalize_F(F_rank2, T1, T2)

        return F_rank2

    def _check_valid_pose(self, R_base, T_base, R_candidate, T_candidate):
        for i in range(len(self.pts1)):
            x1 = np.array([self.pts1[i, 0], self.pts1[i, 1], 1])  # Homogeneous coordinates for pts1
            x2 = np.array([self.pts2[i, 0], self.pts2[i, 1], 1])  # Homogeneous coordinates for pts2

            P1 = CameraPose.calculate_projection_matrix(R_base, T_base)
            P2 = CameraPose.calculate_projection_matrix(R_candidate, T_candidate)

            # Triangulate the 3D point from both images
            X = CameraPose.triangulate_point(x1, x2, P1, P2)

            # Check if the depth (Z) of the triangulated point is positive in both views
            if X[2] < 0:
                return False

        return True

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
        X = Vt[-1]
        X /= X[3]  # Normalize to get homogeneous coordinates (X, Y, Z, 1)

        return X[:3]  # Return 3D point (X, Y, Z)

    @staticmethod
    def calculate_projection_matrix(R, t):
        return np.hstack([R, t.reshape(-1, 1)])

    @staticmethod
    def solve_pnp(pts_3d, pts_2d, K, dist_coeffs=None):
        # Ensure points are float32
        pts_3d = np.asarray(pts_3d, dtype=np.float32)
        pts_2d = np.asarray(pts_2d, dtype=np.float32)

        if pts_3d.shape[0] < 4 or pts_2d.shape[0] < 4:
            return None, None

        # SolvePnP with RANSAC to handle outliers
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=pts_3d,  # 3D points in world space
            imagePoints=pts_2d,  # 2D points in image space
            cameraMatrix=K,  # Intrinsic camera matrix
            distCoeffs=dist_coeffs,  # Distortion coefficients (None if undistorted)
            flags=cv2.SOLVEPNP_ITERATIVE  # Standard iterative approach
        )

        if not success:
            return None, None

        # Back to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        return R, tvec


class BundleAdjustment:
    def __init__(self, num_cameras, num_points, camera_indices, point_indices, points_2d, camera_params, points_3d, K_list):
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.camera_indices = camera_indices
        self.point_indices = point_indices
        self.points_2d = points_2d
        self.camera_params = camera_params
        self.points_3d = points_3d
        self.K_list = K_list

    def sparse_bundle_adjustment(self):
        # Flatten the initial parameters
        initial_params = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))

        # Optimize
        result = least_squares(
            self.compute_residuals,
            initial_params,
            args=(self.num_cameras, self.num_points, self.camera_indices, self.point_indices, self.points_2d, self.K_list),
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
        return point_proj[:2] / point_proj[2]

    def compute_residuals(self, params, num_cameras, num_points, camera_indices, point_indices, points_2d, K_list):
        # 6 params in total for each camera, 3 for Rotation, 3 for translation
        camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
        points_3d = params[num_cameras * 6:].reshape((num_points, 3))
        residuals = []

        for cam_idx, point_idx, observed_2d in zip(camera_indices, point_indices, points_2d):
            # Extract camera parameters
            R_vec = camera_params[cam_idx, :3]
            t = camera_params[cam_idx, 3:6]

            # Convert Rotation Rodrigues vector to rotation matrix
            R, _ = cv2.Rodrigues(R_vec)

            # Reproject the 3D point
            point_3d = points_3d[point_idx]
            K = K_list[cam_idx]
            projected_2d = self.project_point(point_3d, R, t, K)

            # Compute residual (reprojection error)
            residuals.append(projected_2d - observed_2d)

        return np.concatenate(residuals)
