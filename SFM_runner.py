from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from CV_SFM import OpenCVSFM
from FeatureExtractor.SIFT.ScaleRotInvSIFT import ScaleRotInvSIFT
from FeatureMatcher import NNRatioFeatureMatcher
from SFM import CameraPose, SensorType, Util
from bundle_adjustment import BundleAdjustment
from feature_runner import FeatureRunner
import cv2
from scipy.optimize import least_squares
from utils.utils import _plot_3d_points, _convert_matches_to_coords, _show_correspondence_lines, _save_image

#from utils.image_utils import _convert_matches_to_coords
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class Matches:
    def __init__(self, matches, confidence, p1, p2, K1, K2):
        self.matches = matches
        self.confidence = confidence
        self.p1 = p1
        self.p2 = p2
        self.K1 = K1
        self.K2 = K2

class SFMRunner:
    def __init__(self):
        self.img_path = "test_data/tallneck_mini"

        # Threshold for point tolerance when matching points from prev frame to current to find best next frame
        next_frame_point_dist_thresh = 5.0

        # How many images should be used from img_path, for ease of testing
        max_img = 8

        self.global_poses = []
        self.global_points_3D = []
        self.global_points_2D = []
        self.frame_indices = []  # Stores the frame index for each 2D point
        self.point_indices = []  # Maps 2D points to their corresponding 3D points
        self.global_K = []

        self.best_matches: Tuple | None = None
        self.processed_pairs = set()
        self.processed_frames = []

        self.all_matches: list[list[Matches | None]] = [[None for _ in range(max_img + 1)] for _ in range(max_img + 1)]

        # Concurrent stuff, this processing speed is killing me
        self.lock = Lock()
        self.lock2 = Lock()
        tasks = [(i1, i2) for i1 in range(1, max_img) for i2 in range(1, max_img + 1)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.corner_detect_and_matching_process, i1, i2) for i1, i2 in tasks]

            # Make sure all threads finish and throwing output out in stdout before moving on
            for future in futures:
                future.result()

        # Initial pose
        print("Found best pair {}".format(self.best_matches))

        ransac_max_it = CameraPose.calculate_num_ransac_iterations(0.98, 8, 0.4)
        print("Ransac max iterations {}".format(ransac_max_it))

        # Perform initial ransac to figure out initial structure (baseline)
        initial_matches = self.all_matches[self.best_matches[0]][self.best_matches[1]]
        R1 = np.eye(3)
        t1 = np.zeros(3)

        cam_pose = CameraPose(initial_matches.p1, initial_matches.p2, initial_matches.K1, initial_matches.K2)
        R2, t2, p1, p2 = cam_pose.ransac_camera_motion(R1, t1, max_iterations=ransac_max_it)

        # Triangulate using inliers from ransac
        P1 = CameraPose.calculate_projection_matrix(initial_matches.K1, R1, t1)
        P2 = CameraPose.calculate_projection_matrix(initial_matches.K2, R2, t2)
        p3d = np.array([CameraPose.triangulate_point(pts1, pts2, P1, P2) for pts1, pts2 in zip(p1, p2)])

        R1, _ = cv2.Rodrigues(R1)
        R2, _ = cv2.Rodrigues(R2)

        # Append data for Bundle Adjustment later
        self.global_poses.append((R1, t1))
        self.global_poses.append((R2, t2))

        self.processed_pairs.clear()
        self.processed_pairs.add(self.best_matches)
        self.processed_pairs.add((self.best_matches[1], self.best_matches[0]))
        self.add_points(p3d, p1, 0)
        self.add_points(p3d, p2, 1)

        self.global_K.append(initial_matches.K1)
        self.global_K.append(initial_matches.K2)

        best_prev_frame = None
        best_next_frame = None

        while True:  # For now, loops forever until there's no available next frame
            best_score = 0
            best_pair = None

            # Take second image as the reference image for the next frame, for consistency
            i = self.best_matches[1]

            result_3D = []
            result_2D = []

            # Loop through all its pairs
            for j in range(1, max_img + 1):
                if (i, j) in self.processed_pairs or (self.all_matches[i][j] is None):
                    continue

                matches = self.all_matches[i][j]
                current_2D_points = matches.p2

                # Project global 3D points to the current frame
                projections_2D = Util.project_points_to_image(self.global_points_3D, matches.K2, self.global_poses[i-1])

                for idx, proj_point in enumerate(projections_2D):
                    distances = Util.compute_euclidean_distance(current_2D_points, proj_point[np.newaxis])
                    closest_match_idx = np.argmin(distances)

                    # Check if the match is within the tolerance threshold
                    if distances[closest_match_idx] < next_frame_point_dist_thresh:
                        result_3D.append(self.global_points_3D[idx])
                        result_2D.append(current_2D_points[closest_match_idx])

                if len(result_2D) >= 6 and len(result_2D) > best_score:
                    best_score = len(result_2D)
                    best_prev_frame = np.array(result_3D)
                    best_next_frame = np.array(result_2D)
                    best_pair = (i, j)

            print(f"Found best pair {best_pair}")
            if best_pair is None:
                break

            # Perform PnP with all matched points
            matches = self.all_matches[best_pair[0]][best_pair[1]]
            K = matches.K2
            R3, t3 = CameraPose.solve_pnp(best_prev_frame, best_next_frame, K)

            # Handle invalid pose estimates
            if R3 is None:
                print("Cannot determine Pose")
                continue

            # Triangulate new points
            P1 = CameraPose.calculate_projection_matrix(self.global_K[best_pair[0]], *self.global_poses[best_pair[0]])
            P2 = CameraPose.calculate_projection_matrix(K, R3, t3)
            new_3D_points = np.array([
                CameraPose.triangulate_point(pts1, pts2, P1, P2)
                for pts1, pts2 in zip(matches.p1, matches.p2)
            ])

            # Add new points to the global set
            self.add_points(new_3D_points, matches.p2, len(self.global_poses))

            # Update global pose and intrinsics
            R3, _ = cv2.Rodrigues(R3)
            self.global_poses.append((R3, t3))
            self.global_K.append(K)

            # Mark the pair as processed
            self.processed_pairs.add(best_pair)
            self.processed_pairs.add((best_pair[1], best_pair[0]))

            # Update best match for the next iteration
            self.best_matches = best_pair

        # Bundle adjustments to minimize reprojection errors
        num_cameras, num_points, camera_indices, point_indices, points_2D, camera_params, points_3D, K_list = self.prepare_for_ba()
        ba = BundleAdjustment(
            camera_params=camera_params,
            num_cameras=num_cameras,
            num_points=num_points,
            camera_indices=camera_indices,
            point_indices=point_indices,
            points_2d=points_2D,
            points_3d=points_3D,
            K_list=K_list)
        optimized_camera_params, optimized_points_3D = ba.sparse_bundle_adjustment()
        self.global_points_3D = optimized_points_3D.tolist()

        print("Num 3D points: ", len(self.global_points_3D))

        # Visualization
        self.visualize_3d_points_and_cameras()

    def visualize_3d_points_and_cameras(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D points
        points_3d = np.array(self.global_points_3D)
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=0.5, label='3D Points')

        print("Num 3D points: ", len(points_3d))
        print("Num cameras: ", len(self.global_poses))

        # Plot camera poses
        for i, (R, t) in enumerate(self.global_poses):
            R, _ = cv2.Rodrigues(R)
            camera_position = -R.T @ t
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='red', label=f'Camera {i+1}')

        # Set labels and legends
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Points and Camera Positions')
        ax.legend()
        plt.show(block=True)

    def add_points(self, points_3d, points_2d, frame_idx):
        for i, (p3d, p2d) in enumerate(zip(points_3d, points_2d)):
            if self.is_new_point(p3d):
                self.global_points_3D.append(p3d)
                point_idx = len(self.global_points_3D) - 1
            else:
                point_idx = self.find_existing_point(p3d)

            self.global_points_2D.append(p2d)
            self.frame_indices.append(frame_idx)
            self.point_indices.append(point_idx)

    def is_new_point(self, p3d, threshold=1e-3):
        if not self.global_points_3D:
            return True

        distances = Util.compute_euclidean_distance(np.array(self.global_points_3D), p3d[np.newaxis])
        return np.min(distances) >= threshold

    def find_existing_point(self, p3d, threshold=1e-3):
        distances = Util.compute_euclidean_distance(np.array(self.global_points_3D), p3d[np.newaxis])
        min_idx = np.argmin(distances)
        if distances[min_idx] < threshold:
            return min_idx
        raise ValueError("Point not found.")

    def prepare_for_ba(self):
        num_cameras = len(set(self.frame_indices))
        num_points = len(self.global_points_3D)
        camera_indices = self.frame_indices
        point_indices = self.point_indices
        points_2D = np.array(self.global_points_2D)
        points_3D = np.array(self.global_points_3D)
        K_list = np.array(self.global_K)

        print("Num cameras: ", num_cameras)
        print("Camera params: ", len(self.global_poses))

        camera_params = []
        for pose in self.global_poses:
            camera_params.append(np.hstack((pose[0].flatten(), pose[1].flatten())))
        camera_params = np.array(camera_params)

        return num_cameras, num_points, camera_indices, point_indices, points_2D, camera_params, points_3D, K_list

    def corner_detect_and_matching_process(self, i1, i2):
        with self.lock:
            if i1 == i2 or (i1, i2) in self.processed_pairs or (i2, i1) in self.processed_pairs:
                return

            self.processed_pairs.add((i1, i2))

        print("Processing pair {} {}".format(i1, i2))

        K1 = Util.construct_K("{}/{}.jpg".format(self.img_path, i1), SensorType.CROP_FRAME)
        K2 = Util.construct_K("{}/{}.jpg".format(self.img_path, i2), SensorType.CROP_FRAME)

        extractor_params = {
            'num_interest_points': 3000,
            'ksize': 2,
            'gaussian_size': 6,
            'sigma': 1,
            'alpha': 0.02,
            'feature_width': 32,
            'pyramid_level': 8,
            'pyramid_scale_factor': 1.2
        }

        srunner = FeatureRunner("{}/{}.jpg".format(self.img_path, i1), "{}/{}.jpg".format(self.img_path, i2),
                                feature_extractor_class=ScaleRotInvSIFT, extractor_params=extractor_params)
        p1, p2 = _convert_matches_to_coords(srunner.matches, srunner.X1, srunner.Y1, srunner.X2, srunner.Y2, 2500)

        with self.lock2:
            self.all_matches[i1][i2] = Matches(srunner.matches, srunner.confidences, p1, p2, K1, K2)
            self.all_matches[i2][i1] = Matches(srunner.matches, srunner.confidences, p2, p1, K2, K1)

            if self.best_matches is None or srunner.matches.shape[0] > self.all_matches[self.best_matches[0]][self.best_matches[1]].matches.shape[0]:
                self.best_matches = (i1, i2)

class OpenCVSFMRunner:
    def __init__(self, image_paths, K):
        self.image_paths = image_paths
        self.K = K
        self.sfm = OpenCVSFM(K)

    def run(self):
        for i in range(len(self.image_paths) - 1):
            print(f"Processing image pair {i + 1} and {i + 2}...")

            # Load images
            # img1 = cv2.imread(self.image_paths[i])
            # img2 = cv2.imread(self.image_paths[i + 1])

            extractor_params = {
                'num_interest_points': 2500,
                'ksize': 4,
                'gaussian_size': 3,
                'sigma': 3,
                'alpha': 0.05,
                'feature_width': 8,
                'pyramid_level': 3,
                'pyramid_scale_factor': 1.2
            }

            feature_runner = FeatureRunner(
                im1_path=self.image_paths[i],
                im2_path=self.image_paths[i + 1],
                feature_extractor_class=ScaleRotInvSIFT,
                extractor_params=extractor_params,
                print_matches=True,
                output_path=f'output/matches_{i}.jpg'
            )

            # Extract matched points
            X1 = np.array([feature_runner.X1[match[0]] for match in feature_runner.matches])
            Y1 = np.array([feature_runner.Y1[match[0]] for match in feature_runner.matches])
            X2 = np.array([feature_runner.X2[match[1]] for match in feature_runner.matches])
            Y2 = np.array([feature_runner.Y2[match[1]] for match in feature_runner.matches])
            pts1 = np.column_stack((X1, Y1))
            pts2 = np.column_stack((X2, Y2))

            # Compute essential matrix
            E, mask = self.sfm.compute_essential_matrix(pts1, pts2)
            if mask is None:
                print("Essential matrix not found")
                continue
            if E is None or E.shape != (3, 3):
                print("Invalid essential matrix computed:\n", E)
                continue
            pts1_inliers = pts1[mask.ravel() == 1]
            pts2_inliers = pts2[mask.ravel() == 1]

            print(f"pts1 shape: {pts1.shape}, pts2 shape: {pts2.shape}")
            assert pts1.shape[0] >= 5, "Not enough points for computing essential matrix."
            assert pts1.shape == pts2.shape, "Point sets do not match in shape."

            # Recover pose
            R, t, pose_mask = self.sfm.recover_pose(E, pts1_inliers, pts2_inliers)
            pose_mask = pose_mask.ravel()  # Flatten the mask
            pts1_inliers = pts1_inliers[pose_mask == 255]
            pts2_inliers = pts2_inliers[pose_mask == 255]

            # Triangulate points
            P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = self.K @ np.hstack((R, t))
            points_3D = np.array([
                self.sfm.triangulate_point(x1, x2, P1, P2)
                for x1, x2 in zip(pts1_inliers, pts2_inliers)
            ])

            # Filter triangulated points
            valid_mask = np.isfinite(points_3D).all(axis=1) & (points_3D[:, 2] > 0)
            points_3D_inliers = points_3D[valid_mask]

            # Update global map
            self.sfm.add_pose(R, t)
            self.sfm.add_points(points_3D_inliers)

        self.visualize_results()

    def visualize_results(self):
        """Visualize 3D points and camera poses."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D points
        points = np.array(self.sfm.points_3D)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=0.5, label='3D Points')

        # Plot camera poses
        for i, (R, t) in enumerate(self.sfm.poses):
            camera_position = -R.T @ t
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='red', label=f'Camera {i+1}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def find_corresponding_points(self, points_3D, points_2D, matches_2D, distance_thresh):
        """Match existing 3D points to 2D points in the current frame."""
        matched_3D = []
        matched_2D = []

        for i, point_2D in enumerate(matches_2D):
            distances = Util.compute_euclidean_distance(points_2D, point_2D[np.newaxis])
            closest_idx = np.argmin(distances)

            if distances[closest_idx] < distance_thresh:
                matched_3D.append(points_3D[closest_idx])
                matched_2D.append(point_2D)

        return np.array(matched_3D), np.array(matched_2D)

# class SFMRunner:
#     def __init__(self, extractor_params: dict = {}):
#         self.global_camera_poses = [(np.eye(3), np.zeros(3))]
#         self.global_points_3d = []
#         self.point_observations = []
#
#         self._process_images(extractor_params)
#
#     def _process_images(self, extractor_params: dict):
#         """Processes image pairs to calculate camera poses and 3D points."""
#         # for i1 in range(1, 6):
#         #     for i2 in range(1, 7):
#         for i1 in range(1, 3):
#             for i2 in range(1, 3):
#                 if i1 == i2:
#                     continue
#
#                 srunner = FeatureRunner(
#                     f"test_data/tallneck_mini/{i1}.jpg",
#                     f"test_data/tallneck_mini/{i2}.jpg",
#                     feature_extractor_class=ScaleRotInvSIFT,
#                     extractor_params=extractor_params
#                 )
#
#                 if len(srunner.matches) < 40:
#                     print(f"{i1} {i2}: Not enough matches")
#                     continue
#
#                 R, T = self._calculate_camera_pose(srunner)
#                 if R is None:
#                     print(f"{i1} {i2}: No valid configuration")
#                     continue
#
#                 self._update_camera_poses_and_points(srunner, R, T)
#
#         K = np.eye(3)
#
#         print("Performing bundle adjustment...")
#         self.bundle_adjustment(K)
#
#         self._plot_results()
#
#     def _calculate_camera_pose(self, srunner: FeatureRunner):
#         """Calculates the camera pose using RANSAC."""
#         p1, p2 = _convert_matches_to_coords(
#             srunner.matches, srunner.X1, srunner.Y1, srunner.X2, srunner.Y2, num_matches=2500
#         )
#         cam_pose = CameraPose(p1, p2)
#         return cam_pose.ransac_camera_motion()
#
#     def _update_camera_poses_and_points(self, srunner: FeatureRunner, R, T):
#         """Updates the global camera poses and triangulates new points."""
#         R1, T1 = self.global_camera_poses[-1]
#         self.global_camera_poses.append((R, T))
#
#         p1, p2 = _convert_matches_to_coords(
#             srunner.matches, srunner.X1, srunner.Y1, srunner.X2, srunner.Y2, num_matches=2500
#         )
#         new_points_3d = [
#             CameraPose.triangulate_point(pts1, pts2, R1, T1, R, T)
#             for pts1, pts2 in zip(p1, p2)
#         ]
#
#         for idx, (point_3d, point_2d) in enumerate(zip(new_points_3d, p1)):
#             if point_3d is not None:
#                 self.global_points_3d.append(point_3d)
#                 self.point_observations.append((len(self.global_camera_poses) - 1, idx, point_2d))
#
#     def _plot_results(self):
#         """Plots the 3D points."""
#         _plot_3d_points(self.global_points_3d)
#
#     def bundle_adjustment(self, K):
#         n_cameras = len(self.global_camera_poses)
#         n_points = len(self.global_points_3d)
#
#         # Flatten camera poses and 3D points
#         camera_params = []
#         for R, T in self.global_camera_poses:
#             rvec, _ = cv2.Rodrigues(R)  # Convert rotation matrix to rotation vector
#             camera_params.append(np.hstack([rvec.flatten(), T.flatten()]))
#         camera_params = np.array(camera_params).ravel()
#         points_3d = np.array(self.global_points_3d).ravel()
#
#         # Prepare observation indices and 2D points
#         camera_indices = []
#         point_indices = []
#         points_2d = []
#         for cam_idx, pt_idx, pt_2d in self.point_observations:
#             camera_indices.append(cam_idx)
#             point_indices.append(pt_idx)
#             points_2d.append(pt_2d)
#         camera_indices = np.array(camera_indices)
#         point_indices = np.array(point_indices)
#         points_2d = np.array(points_2d)
#
#         # Define the reprojection error function
#         def reprojection_error(params):
#             camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
#             points_3d = params[n_cameras * 6:].reshape((n_points, 3))
#             errors = []
#
#             for cam_idx, pt_idx, pt_2d in zip(camera_indices, point_indices, points_2d):
#                 rvec = camera_params[cam_idx, :3]
#                 tvec = camera_params[cam_idx, 3:]
#                 X_world = points_3d[pt_idx]
#
#                 R, _ = cv2.Rodrigues(rvec)
#                 try:
#                     print(f"R shape: {R.shape}, X_world shape: {X_world.shape}, tvec shape: {tvec.shape}")
#                     print(f"R: {R}, Determinant: {np.linalg.det(R)}")
#                     print(
#                         f"R finite: {np.isfinite(R).all()}, X_world finite: {np.isfinite(X_world).all()}, tvec finite: {np.isfinite(tvec).all()}")
#                     X_cam = R @ X_world + tvec
#                 except Exception as e:
#                     print(f"Error at X_cam computation: {e}")
#                 X_proj = K @ X_cam
#                 X_proj /= X_proj[2]  # Normalize to homogeneous
#
#                 errors.append(np.linalg.norm(X_proj[:2] - pt_2d))
#
#             return np.array(errors)
#
#         # Perform optimization
#         x0 = np.hstack([camera_params, points_3d])
#         res = least_squares(reprojection_error, x0)
#
#         # Extract optimized parameters
#         optimized_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
#         optimized_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
#
#         # Update global variables
#         self.global_camera_poses = [
#             (cv2.Rodrigues(optimized_camera_params[i, :3])[0], optimized_camera_params[i, 3:])
#             for i in range(n_cameras)
#         ]
#         self.global_points_3d = optimized_points_3d.tolist()
#
#     def _prepare_camera_params(self):
#         """Prepares the initial camera parameters."""
#         return [
#             (cv2.Rodrigues(R)[0], T)
#             for R, T in self.global_camera_poses
#         ]
#
#     def _prepare_observations(self):
#         """Prepares observation indices and 2D points."""
#         camera_indices, point_indices, points_2d = [], [], []
#         for cam_idx, pt_idx, pt_2d in self.point_observations:
#             camera_indices.append(cam_idx)
#             point_indices.append(pt_idx)
#             points_2d.append(pt_2d)
#
#         return np.array(camera_indices), np.array(point_indices), np.array(points_2d)
#
#     def _update_global_camera_poses(self, rvecs, tvecs):
#         """Updates the global camera poses after bundle adjustment."""
#         self.global_camera_poses = [
#             (cv2.Rodrigues(rvec)[0], tvec)
#             for rvec, tvec in zip(rvecs, tvecs)
#         ]
