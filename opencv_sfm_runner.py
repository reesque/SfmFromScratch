
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
