import cv2
import numpy as np
from numpy.linalg import svd
from PIL import Image
from PIL.ExifTags import TAGS
from enum import Enum
from scipy.optimize import least_squares


class SensorType(Enum):
    """
    Type of camera sensor
    """
    MEDIUM_FORMAT = 1
    FULL_FRAME = 2
    CROP_FRAME = 3
    MICRO_FOUR_THIRD = 4
    ONE_INCH = 5
    SMARTPHONE = 6


class CameraPose:
    def __init__(self, pts1, pts2, K1, K2):
        """
        Initial camera pose estimation. Also contains helper functions related to
        camera pose, such as triangulation, construct K, find inliers, etc.

        :param pts1: Correspondence of image 1
        :param pts2: Correspondence of image 2
        :param K1: Camera intrinsic of image 1
        :param K2: Camera intrinsic of image 2
        """
        self.pts1 = pts1
        self.pts2 = pts2
        self.K1 = K1
        self.K2 = K2

    def ransac_camera_motion(self, R_base, T_base, threshold=1.0, max_iterations=1000):
        best_inliers1, best_inliers2 = [], []
        best_r, best_t = None, None

        if len(self.pts1) < 8:
            return None, None, None, None

        np.random.seed(5)

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

            if np.linalg.det(R1) < 0:
                R1 = R1 * -1

            if np.linalg.det(R2) < 0:
                R2 = R2 * -1

            # Translation vector is the last column of U
            T = U[:, 2]

            # Compute the projection error for all four combinations of R and T
            for R_candidate, T_candidate in [(R1, T), (R1, -T), (R2, T), (R2, -T)]:
                if not self._check_valid_pose(R_base, T_base, R_candidate, T_candidate):
                    continue  # Skip invalid pose

                # Homogeneous
                sample_a_homogeneous = np.column_stack((self.pts1, np.ones(len(self.pts1))))
                sample_b_homogeneous = np.column_stack((self.pts2, np.ones(len(self.pts2))))

                # Epipolar line of point b: l_2 = F.p_1 due to epipolar constraint
                lb = (F @ sample_a_homogeneous.T).T

                # Dist is |lb . p2| / norm(lb)
                distances = np.abs(np.sum(lb * sample_b_homogeneous, axis=1)) / np.sqrt(lb[:, 0] ** 2 + lb[:, 1] ** 2)

                inlier_mask = distances < threshold

                # If this rotation and translation give more inliers, select it as the best
                if np.sum(inlier_mask) > len(best_inliers1):
                    best_inliers1 = self.pts1[inlier_mask]
                    best_inliers2 = self.pts2[inlier_mask]
                    best_r, best_t = R_candidate, T_candidate

        return best_r, best_t, np.array(best_inliers1), np.array(best_inliers2)

    def _check_valid_pose(self, R_base, T_base, R_candidate, T_candidate):
        for i in range(len(self.pts1)):
            x1 = np.array([self.pts1[i, 0], self.pts1[i, 1], 1])  # Homogeneous coordinates for pts1
            x2 = np.array([self.pts2[i, 0], self.pts2[i, 1], 1])  # Homogeneous coordinates for pts2

            P1 = CameraPose.calculate_projection_matrix(R_base, T_base, self.K1)
            P2 = CameraPose.calculate_projection_matrix(R_candidate, T_candidate, self.K2)

            # Triangulate the 3D point from both images
            X = CameraPose.triangulate_point(x1, x2, P1, P2)

            # Convert X to camera coordinates of both views
            X_base = R_base @ X[:3] + T_base
            X_candidate = R_candidate @ X[:3] + T_candidate

            # Check if the depth (Z) of the triangulated point is positive in both views
            if X_base[2] < 1e-6 or X_candidate[2] < 1e-6:
                return False

        return True

    @staticmethod
    def find_inliers(p1, p2, threshold=1.0, max_iterations=1000):
        best_inliers1, best_inliers2 = [], []

        if len(p1) < 8:
            return None, None, None, None

        np.random.seed(5)

        for _ in range(max_iterations):
            # Randomly sample 8 correspondences
            indices = np.random.choice(len(p1), 8, replace=False)
            sample_points1 = p1[indices]
            sample_points2 = p2[indices]

            # Compute the fundamental matrix using the 8-point algorithm
            F = CameraPose._compute_fundamental_matrix(sample_points1, sample_points2)

            sample_a_homogeneous = np.column_stack((p1, np.ones(len(p1))))
            sample_b_homogeneous = np.column_stack((p2, np.ones(len(p2))))

            # Epipolar line of point b: l_2 = F.p_1 due to epipolar constraint
            lb = (F @ sample_a_homogeneous.T).T

            # Dist is |lb . p2| / norm(lb)
            distances = np.abs(np.sum(lb * sample_b_homogeneous, axis=1)) / np.sqrt(lb[:, 0] ** 2 + lb[:, 1] ** 2)

            inlier_mask = distances < threshold

            # If this rotation and translation give more inliers, select it as the best
            if np.sum(inlier_mask) > len(best_inliers1):
                best_inliers1 = p1[inlier_mask]
                best_inliers2 = p2[inlier_mask]

        return np.array(best_inliers1), np.array(best_inliers2)

    @staticmethod
    def normalize_points(points):
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

    @staticmethod
    def unnormalize_F(F_norm, T_a, T_b):
        return T_b.T @ F_norm @ T_a

    @staticmethod
    def calculate_num_ransac_iterations(prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
        num_samples = np.log(1 - prob_success) / np.log(1 - (ind_prob_correct ** sample_size))
        return int(num_samples)

    @staticmethod
    def _compute_fundamental_matrix(p1, p2) -> np.ndarray:
        # Ensure points are in homogeneous coordinates
        n = p1.shape[0]
        pts1_hom, T1 = CameraPose.normalize_points(np.hstack([p1, np.ones((n, 1))]))
        pts2_hom, T2 = CameraPose.normalize_points(np.hstack([p2, np.ones((n, 1))]))

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

        F_rank2 = CameraPose.unnormalize_F(F_rank2, T1, T2)

        return F_rank2

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
    def non_linear_triangulation(p3d, p1, p2, P1, P2):
        def global_reprojection_error(flat_p3d, p1, p2, P1, P2):
            # Reshape the flat array into a 3D points array
            p3d = flat_p3d.reshape(-1, 3)

            errors = []
            for i, (pts1, pts2) in enumerate(zip(p1, p2)):
                X = np.hstack([p3d[i], 1])  # Convert to homogeneous coordinates
                x1_reprojected = P1 @ X
                x2_reprojected = P2 @ X
                x1_reprojected /= x1_reprojected[2]  # Normalize
                x2_reprojected /= x2_reprojected[2]  # Normalize

                # Calculate reprojection error for both cameras
                error1 = pts1 - x1_reprojected[:2]
                error2 = pts2 - x2_reprojected[:2]
                errors.extend(error1)
                errors.extend(error2)

            return np.array(errors)

        flat_p3d_initial = p3d.reshape(-1)

        # Minimize the global reprojection error
        result = least_squares(
            global_reprojection_error,
            flat_p3d_initial,
            args=(p1, p2, P1, P2),
            method='lm'  # Levenberg-Marquardt
        )

        # Reshape the optimized 3D points back to original shape
        optimized_p3d = result.x.reshape(-1, 3)
        return optimized_p3d

    @staticmethod
    def triangulate_points(x1, x2, P1, P2):
        n = x1.shape[0]
        pts1_hom, T1 = CameraPose.normalize_points(np.hstack([x1, np.ones((n, 1))]))
        pts2_hom, T2 = CameraPose.normalize_points(np.hstack([x2, np.ones((n, 1))]))

        P1_normalized = T1 @ P1
        P2_normalized = T2 @ P2

        p3d = np.array([CameraPose.triangulate_point(pts1, pts2, P1_normalized, P2_normalized) for pts1, pts2 in zip(pts1_hom, pts2_hom)])

        # Convert back to Euclidean coordinates
        p3d = p3d[:, :3]

        return p3d

    @staticmethod
    def calculate_projection_matrix(R, t, K):
        return K @ np.hstack([R, t.reshape(-1, 1)])

    @staticmethod
    def construct_K(image_path, sensor_type: SensorType):
        """
        Calculating camera intrinsic using EXIF data, providing with the sensor type of the camera
        
        :param image_path: Path of the image
        :param sensor_type: Type of sensor of the camera
        :return: K
        """
        image = Image.open(image_path)
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

        if sensor_type is SensorType.MEDIUM_FORMAT:
            sensor_width = 53.0
            sensor_height = 40.20
        elif sensor_type is SensorType.FULL_FRAME:
            sensor_width = 35.0
            sensor_height = 24.0
        elif sensor_type is SensorType.CROP_FRAME:
            sensor_width = 23.6
            sensor_height = 15.60
        elif sensor_type is SensorType.MICRO_FOUR_THIRD:
            sensor_width = 17.0
            sensor_height = 13.0
        elif sensor_type is SensorType.ONE_INCH:
            sensor_width = 12.80
            sensor_height = 9.60
        elif sensor_type is SensorType.SMARTPHONE:
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

    @staticmethod
    def project_point(point_3d, R, t, K):
        if R.shape == (3,):
            R = cv2.Rodrigues(R)[0]

        point_3d_h = np.append(point_3d, 1)
        P = CameraPose.calculate_projection_matrix(R, t, K)
        point_proj = P @ point_3d_h
        return point_proj[:2] / point_proj[2]

    @staticmethod
    def compute_reprojection_error(points_3d, points_2d, R, t, K):
        # Project 3D points onto the image plane
        projected_points = np.array([CameraPose.project_point(p3d, R, t, K) for p3d in points_3d])

        # Compute the error
        errors = CameraPose.compute_euclidean_distance(points_2d, projected_points)
        mean_error = np.mean(errors)
        return mean_error


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
            ftol=1e-6,
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
            projected_2d = CameraPose.project_point(point_3d, R, t, K)

            # Compute residual (reprojection error)
            residuals.append(projected_2d - observed_2d)

        return np.concatenate(residuals)
