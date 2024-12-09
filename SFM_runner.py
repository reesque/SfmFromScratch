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
from utils.utils import _plot_3d_points, _convert_matches_to_coords, _show_correspondence_lines, _save_image, \
    get_matches, _load_image, _show_correspondence2

#from utils.image_utils import _convert_matches_to_coords
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class Matches:
    def __init__(self, matches, confidence, p1, p2, F, mask):
        self.matches = matches
        self.confidence = confidence
        self.p1 = p1
        self.p2 = p2
        # self.K1 = K1
        # self.K2 = K2
        self.F = F
        self.mask = mask

class SFMRunner:
    def __init__(self):
        self.img_path = "test_data/fountain_mini"
        self.next_frame_point_dist_thresh = 5.0
        self.max_img = 2

        # Global structures for poses, points, and intrinsics
        self.global_poses = []  # Stores (R, t) for each frame
        self.global_points_3D = []  # Point cloud
        # self.global_K = None # [None for _ in range(self.max_img)]  # Intrinsics for each frame
        self.K = np.array([
            [2759.48, 0, 1520.69],
            [0, 2764.16, 1006.81],
            [0, 0, 1]
        ])
        # Util.construct_K("{}/{}.jpg".format(self.img_path, 1), SensorType.CROP_FRAME)
        self.global_points_2D = []
        self.frame_indices = []
        self.point_indices = []

        self.processed_pairs = set()
        self.prev_matches = []

        self.all_matches = [[None for _ in range(self.max_img + 1)] for _ in range(self.max_img + 1)]

        # Multi-threaded matching setup
        self.lock = Lock()
        self.lock2 = Lock()
        tasks = [(i1, i2) for i1 in range(1, self.max_img) for i2 in range(1, self.max_img + 1)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.corner_detect_and_matching_process, i1, i2) for i1, i2 in tasks]

            for future in futures:
                future.result()

        # Perform initial pose estimation and triangulation
        self.initialize_structure()

        # Expand the structure by registering new views and triangulating new points
        self.expand_structure()

        # Final bundle adjustment
        # self.optimize_bundle()

        # Visualization
        self.visualize_3d_points_and_cameras()

    def initialize_structure(self):
        matches = self.all_matches[1][2]

        # Estimate Fundamental Matrix and recover pose
        F, mask = cv2.findFundamentalMat(matches.p1, matches.p2, cv2.FM_RANSAC, 1.0, 0.98)
        inlier_p1 = matches.p1[mask.ravel() == 1]
        inlier_p2 = matches.p2[mask.ravel() == 1]

        print(f"Inlier matches: {len(inlier_p1)}")
        E = self.K.T @ F @ self.K
        _, R2, t2, _ = cv2.recoverPose(E, inlier_p1, inlier_p2, self.K)

        # Convert 2D points to homogeneous coordinates
        inlier_p1_hom = np.hstack((inlier_p1, np.ones((len(inlier_p1), 1))))
        inlier_p2_hom = np.hstack((inlier_p2, np.ones((len(inlier_p2), 1))))

        # Normalize 2D points
        normalized_p1 = (np.linalg.inv(self.K) @ inlier_p1_hom.T).T
        normalized_p2 = (np.linalg.inv(self.K) @ inlier_p2_hom.T).T

        # Convert 2D Homogeneous to 2D
        normalized_p1 = cv2.convertPointsFromHomogeneous(normalized_p1)[:, 0, :]
        normalized_p2 = cv2.convertPointsFromHomogeneous(normalized_p2)[:, 0, :]

        # Triangulate initial 3D points
        P1 = CameraPose.calculate_projection_matrix(self.K, np.eye(3), np.zeros(3))
        P2 = CameraPose.calculate_projection_matrix(self.K, R2, t2)

        points_4D = cv2.triangulatePoints(P1, P2, normalized_p1.T, normalized_p2.T)
        points_3D = cv2.convertPointsFromHomogeneous(points_4D.T)[:, 0, :]

        print(f"Found {len(points_3D)} 2D-3D matches for initial pair.")

        # Store initial poses, points, and intrinsics
        self.global_poses.extend([(np.eye(3), np.zeros(3)), (R2, t2.flatten())])

        # Store 2D points and frame indices
        self.add_points(points_3D, inlier_p1, 0)
        self.add_points(points_3D, inlier_p2, 1)

        self.processed_pairs.add((1, 2))

    def expand_structure(self):
        """
        Iteratively expand the structure by registering new views and triangulating new points.
        """
        for new_view_idx in range(3, self.max_img+1):
            print(f"Processing view {new_view_idx}...")

            # Step 1: Find 2D-3D Matches for Pose Estimation
            points_3D, points_2D = self.find_2D_3D_correspondences(new_view_idx)
            if len(points_3D) < 6:
                print(f"Found only {len(points_3D)} 2D-3D matches for view {new_view_idx}. Skipping...")
                continue

            points_2D_normalized = cv2.undistortPoints(points_2D[:, np.newaxis], self.K, None)

            # Step 2: Estimate Pose Using PnP
            success, R, t, inliers = cv2.solvePnPRansac(
                points_3D[:, np.newaxis], points_2D_normalized, self.K, None,
                reprojectionError=8.0, iterationsCount=1000, confidence=0.999
            )
            if not success:
                print(f"Pose estimation failed for view {new_view_idx}. Skipping...")
                continue

            R, _ = cv2.Rodrigues(R)

            # Debug Camera Poses
            print(f"Camera {new_view_idx} Pose:\nR = {R}\nt = {t.flatten()}")

            # Check baseline
            prev_camera_center = -self.global_poses[-1][0].T @ self.global_poses[-1][1]
            new_camera_center = -R.T @ t
            baseline = np.linalg.norm(new_camera_center - prev_camera_center)
            print(f"Baseline between previous and new camera: {baseline:.2f}")
            if baseline < 0.01:  # Adjust this threshold
                print(f"Baseline too small for triangulation. Skipping view {new_view_idx}...")
                #continue

            projected_points, _ = cv2.projectPoints(points_3D[inliers[:, 0]], R, t, self.K, None)
            errors = np.linalg.norm(points_2D[inliers[:, 0]] - projected_points.squeeze(), axis=1)
            mean_error = np.mean(errors)
            print(f"Reprojection error for view {new_view_idx}: {mean_error:.2f}")
            if mean_error > 5.0:
                print(f"High reprojection error for view {new_view_idx}. Skipping...")
                # continue

            new_points = np.array(points_3D)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c='orange', label='3D Points (before triangulation')
            ax.legend()
            plt.show()

            # Step 3: Triangulate Points Between the New View and Previous Views
            new_3D_points = self.triangulate_new_points(new_view_idx, R, t)

            new_points = np.array(new_3D_points)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c='orange', label='New Points')
            ax.legend()
            plt.show()

            # Step 4: Update Global Structures
            self.global_poses.append((R, t.flatten()))
            self.add_points(new_3D_points, inliers, new_view_idx)

            print(f"View {new_view_idx} processed. Total 3D points: {len(self.global_points_3D)}")

    def find_2D_3D_correspondences(self, new_view_idx):
        """
        Find 2D-3D correspondences for pose estimation by matching features
        from the new view to existing 3D points.
        """
        points_3D, points_2D = [], []

        # Iterate over all previous views
        for prev_view_idx in range(1, new_view_idx):
            # Skip if the previous view does not have a valid pose
            if prev_view_idx > len(self.global_poses) or self.global_poses[prev_view_idx-1] is None:
                continue

            # Get matches between the previous view and the new view
            matches = self.all_matches[prev_view_idx][new_view_idx]

            if matches is None:
                continue

            # Project 3D points onto the image plane of the current frame
            projections = Util.project_points_to_image(
                np.array(self.global_points_3D), self.K, self.global_poses[prev_view_idx - 1]
            )
            distances = Util.compute_euclidean_distance(matches.p2, projections)

            # Filter inliers
            F, mask = matches.F, matches.mask
            inlier_p2 = matches.p2[mask.ravel() == 1]
            distances = distances[mask.ravel() == 1]

            # Find closest 3D point for each 2D point
            closest_indices = np.argmin(distances, axis=1)
            closest_distances = distances[np.arange(distances.shape[0]), closest_indices]

            # Identify valid matches within the threshold
            valid_matches = closest_distances < self.next_frame_point_dist_thresh

            # Add valid matches to the lists
            for i, valid in enumerate(valid_matches):
                if valid:
                    points_3D.append(self.global_points_3D[closest_indices[i]])
                    points_2D.append(inlier_p2[i])

        return np.array(points_3D), np.array(points_2D)

    def triangulate_new_points(self, new_view_idx, R, t):
        """
        Triangulate new 3D points between the new view and previous views.
        """
        new_3D_points = []
        for prev_view_idx in range(len(self.global_poses)):
            matches = self.all_matches[prev_view_idx + 1][new_view_idx]
            if matches is None:
                continue

            # Triangulate points
            P1 = CameraPose.calculate_projection_matrix(self.K, *self.global_poses[prev_view_idx])
            P2 = CameraPose.calculate_projection_matrix(self.K, R, t)

            normalized_p1 = cv2.undistortPoints(matches.p1[:, np.newaxis], self.K, None).squeeze()
            normalized_p2 = cv2.undistortPoints(matches.p2[:, np.newaxis], self.K, None).squeeze()

            # points_4D = cv2.triangulatePoints(P1, P2, matches.p1.T, matches.p2.T)
            points_4D = cv2.triangulatePoints(P1, P2, normalized_p1.T, normalized_p2.T)
            points_3D = cv2.convertPointsFromHomogeneous(points_4D.T)

            print("Num point z above 0: ", np.sum(points_3D[:, 0, 2] > 0))

            # Filter and add valid points
            # valid_points = []
            for point in points_3D:
                # if self.is_new_point(point):
                #    new_3D_points.append(point)
                if point[0, 2] > 0:
                    new_3D_points.append(point)
            # valid_points = np.array(valid_points)
            #
            # new_3D_points.append(valid_points)

        if new_3D_points:
            new_3D_points = np.vstack(new_3D_points)
        else:
            new_3D_points = np.empty((0, 3))

        return new_3D_points

    def optimize_bundle(self):
            num_cameras, num_points, camera_indices, point_indices, points_2D, camera_params, points_3D, K = self.prepare_for_ba()
            ba = BundleAdjustment(
                camera_params=camera_params,
                num_cameras=num_cameras,
                num_points=num_points,
                camera_indices=camera_indices,
                point_indices=point_indices,
                points_2d=points_2D,
                points_3d=points_3D,
                K=K)
            optimized_camera_params, optimized_points_3D = ba.sparse_bundle_adjustment()
            self.global_points_3D = optimized_points_3D.tolist()

    def prepare_for_ba(self):
        num_cameras = len(set(self.frame_indices))
        num_points = len(self.global_points_3D)
        camera_indices = self.frame_indices
        point_indices = self.point_indices
        points_2D = np.array(self.global_points_2D)
        points_3D = np.array(self.global_points_3D)
        K = self.K

        print("Num cameras: ", num_cameras)
        print("Camera params: ", len(self.global_poses))

        camera_params = []
        for pose in self.global_poses:
            R_vec, _ = cv2.Rodrigues(pose[0])
            camera_params.append(np.hstack((R_vec.flatten(), pose[1].flatten())))
        camera_params = np.array(camera_params)

        return num_cameras, num_points, camera_indices, point_indices, points_2D, camera_params, points_3D, K #K_list

    def compute_reprojection_error(self, projections_2D, current_2D_points):
        distances = np.linalg.norm(projections_2D - current_2D_points, axis=1)
        return distances

    def visualize_3d_points_and_cameras(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D points
        points_3d = np.array(self.global_points_3D)
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=0.5, label='3D Points')

        print("Num 3D points: ", len(points_3d))
        print("Num cameras: ", len(self.global_poses))

        # Plot camera positions
        for i, pose in enumerate(self.global_poses):
            R, t = pose  # Extract R (3x3 rotation) and t (3x1 translation)

            # Camera position is -R.T @ t
            camera_position = -R.T @ t

            # Plot the camera position
            ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='red', s=50,
                       label=f'Camera {i}' if i == 0 else "")

            # Optionally, plot camera orientation
            camera_axes = R.T  # Camera axes in world coordinates
            scale = 0.1  # Scale for the axes
            ax.quiver(
                camera_position[0], camera_position[1], camera_position[2],  # Origin
                scale * camera_axes[0, 0], scale * camera_axes[1, 0], scale * camera_axes[2, 0], color='r',
                label='X-axis' if i == 0 else ""
            )
            ax.quiver(
                camera_position[0], camera_position[1], camera_position[2],  # Origin
                scale * camera_axes[0, 1], scale * camera_axes[1, 1], scale * camera_axes[2, 1], color='g',
                label='Y-axis' if i == 0 else ""
            )
            ax.quiver(
                camera_position[0], camera_position[1], camera_position[2],  # Origin
                scale * camera_axes[0, 2], scale * camera_axes[1, 2], scale * camera_axes[2, 2], color='b',
                label='Z-axis' if i == 0 else ""
            )

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

    # def prepare_for_ba(self):
    #     num_cameras = len(set(self.frame_indices))
    #     num_points = len(self.global_points_3D)
    #     camera_indices = self.frame_indices
    #     point_indices = self.point_indices
    #     points_2D = np.array(self.global_points_2D)
    #     points_3D = np.array(self.global_points_3D)
    #     K_list = np.array(self.global_K)
    #
    #     print("Num cameras: ", num_cameras)
    #     print("Camera params: ", len(self.global_poses))
    #
    #     camera_params = []
    #     for pose in self.global_poses:
    #         camera_params.append(np.hstack((pose[0].flatten(), pose[1].flatten())))
    #     camera_params = np.array(camera_params)
    #
    #     return num_cameras, num_points, camera_indices, point_indices, points_2D, camera_params, points_3D, K_list

    def corner_detect_and_matching_process(self, i1, i2):
        with self.lock:
            if i1 == i2 or (i1, i2) in self.processed_pairs or (i2, i1) in self.processed_pairs:
                return

            self.processed_pairs.add((i1, i2))

        print("Processing pair {} {}".format(i1, i2))

        # self.K = Util.construct_K("{}/{}.jpg".format(self.img_path, i1), SensorType.CROP_FRAME)

        # K2 = Util.construct_K("{}/{}.jpg".format(self.img_path, i2), SensorType.CROP_FRAME)

        # with self.lock2:
        #     # Assign K1 to the correct index
        #     if self.global_K[i1-1] is None:
        #         self.global_K[i1-1] = K1
        #     else:
        #         K1 = self.global_K[i1-1]  # Use the existing value for consistency
        #
        #     # Assign K2 to the correct index
        #     if self.global_K[i2-1] is None:
        #         self.global_K[i2-1] = K2
        #     else:
        #         K2 = self.global_K[i2-1]


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

        pic_a = _load_image("{}/{}.jpg".format(self.img_path, i1))[:, :, ::-1]
        pic_b = _load_image("{}/{}.jpg".format(self.img_path, i2))[:, :, ::-1]

        scale_a = 1
        scale_b = 1

        n_feat = 5e4

        pic_a = cv2.resize(pic_a, None, fx=scale_a, fy=scale_a)
        pic_b = cv2.resize(pic_b, None, fx=scale_b, fy=scale_b)

        print("Processing pair {} {}".format(i1, i2))
        print("Loading images {} {}".format("{}/{}.jpg".format(self.img_path, i1), "{}/{}.jpg".format(self.img_path, i2)))

        matches, p1, p2 = get_matches(pic_a, pic_b, int(n_feat))

        # Estimate Fundamental Matrix and recover pose
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.98)
        inlier_p1 = p1[mask.ravel() == 1]
        inlier_p2 = p2[mask.ravel() == 1]

        match_image = _show_correspondence2(pic_a, pic_b,
                                           inlier_p1[:, 0], inlier_p1[:, 1],
                                           inlier_p2[:, 0], inlier_p2[:, 1])
        plt.imsave(f"output/match_image_{i1}_{i2}.png", match_image)

        print(f"Found matches for pair {i1} {i2}: {len(p1)}")

        with self.lock2:
            self.all_matches[i1][i2] = Matches(matches, None, p1, p2, F, mask)
            self.all_matches[i2][i1] = Matches(matches, None, p2, p1, F, mask)

            # if self.best_matches is None or len(matches) > len(self.all_matches[self.best_matches[0]][self.best_matches[1]].matches):
            #     self.best_matches = (i1, i2)
        #
        # srunner = FeatureRunner("{}/{}.jpg".format(self.img_path, i1), "{}/{}.jpg".format(self.img_path, i2),
        #                        feature_extractor_class=ScaleRotInvSIFT, extractor_params=extractor_params)
        # p1, p2 = _convert_matches_to_coords(srunner.matches, srunner.X1, srunner.Y1, srunner.X2, srunner.Y2, 2500)
        #
        # # Estimate Fundamental Matrix and recover pose
        # F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.98)
        #
        # with self.lock2:
        #     self.all_matches[i1][i2] = Matches(srunner.matches, srunner.confidences, p1, p2, K1, K2, F, mask)
        #     self.all_matches[i2][i1] = Matches(srunner.matches, srunner.confidences, p2, p1, K2, K1, F, mask)
        #
        #     if self.best_matches is None or srunner.matches.shape[0] > self.all_matches[self.best_matches[0]][self.best_matches[1]].matches.shape[0]:
        #         self.best_matches = (i1, i2)