import os
import copy
from typing import Tuple

import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
from matplotlib import cm

from FeatureExtractor import FeatureExtractor
from FeatureExtractor.SIFT.ScaleRotInvSIFT import ScaleRotInvSIFT
from FeatureMatcher import NNRatioFeatureMatcher
from PoseEstimator import PoseEstimator
from SFM import *

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from Visualizer import Visualizer


class FeatureRunner:
    def __init__(self, im1_path: str, im2_path: str, scale_factor: float = 0.5,
                 feature_extractor_class: FeatureExtractor = None, extractor_params: dict = {},
                 print_img: bool = False, print_features: bool = False,
                 print_matches: bool = False, outputSuffix="", match_threshold=0.8):
        self.feature_extractor = feature_extractor_class

        if self.feature_extractor is None:
            raise ValueError("Please provide a feature extractor class")

        self.outputSuffix = outputSuffix
        self._image1 = _load_image(im1_path)
        self._image2 = _load_image(im2_path)

        # Rescale images if desired
        self._image1 = _PIL_resize(self._image1,
                                   (int(self._image1.shape[1] * scale_factor),
                                    int(self._image1.shape[0] * scale_factor)))
        self._image2 = _PIL_resize(self._image2,
                                   (int(self._image2.shape[1] * scale_factor),
                                    int(self._image2.shape[0] * scale_factor)))

        # Convert images to grayscale
        self._image1_bw = _rgb2gray(self._image1)
        self._image2_bw = _rgb2gray(self._image2)

        # Initialize feature extractor for each image
        self.extractor1 = self.feature_extractor(self._image1_bw, extractor_params)
        self.extractor2 = self.feature_extractor(self._image2_bw, extractor_params)

        # Extract features
        self.X1, self.Y1 = self.extractor1.detect_keypoints()
        self.descriptors1 = self.extractor1.extract_descriptors()
        self.X2, self.Y2 = self.extractor2.detect_keypoints()
        self.descriptors2 = self.extractor2.extract_descriptors()

        print(f'{len(self.X1)} corners in image 1, {len(self.X2)} corners in image 2')
        print(f'{len(self.descriptors1)} descriptors in image 1, {len(self.descriptors2)} descriptors in image 2')

        # Match features
        self.matcher = NNRatioFeatureMatcher(ratio_threshold=match_threshold)
        self.matches, self.confidences = self.matcher.match_features_ratio_test(self.descriptors1, self.descriptors2)

        print(f'{len(self.matches)} matches found')

        # Optional printing
        if print_img:
            self.print_image()
        if print_features:
            self.print_features()
        if print_matches:
            self.print_matches()

    def print_image(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self._image1)
        plt.subplot(1, 2, 2)
        plt.imshow(self._image2)
        plt.savefig('output/visual{}.png'.format(self.outputSuffix))

    def print_features(self):
        if len(self.X1) == 0 or len(self.X2) == 0:
            print('No interest points to visualize')
            return
        num_pts_to_visualize = 300
        rendered_img1 = _show_interest_points(self._image1, self.X1[:num_pts_to_visualize],
                                              self.Y1[:num_pts_to_visualize])
        rendered_img2 = _show_interest_points(self._image2, self.X2[:num_pts_to_visualize],
                                              self.Y2[:num_pts_to_visualize])

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rendered_img1, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(rendered_img2, cmap='gray')
        plt.savefig('output/features{}.png'.format(self.outputSuffix))

    def print_matches(self):
        if len(self.matches) == 0:
            print('No matches to visualize')
            return
        num_pts_to_visualize = 2500
        c2 = _show_correspondence_lines(
            self._image1,
            self._image2,
            self.X1[self.matches[:num_pts_to_visualize, 0]],
            self.Y1[self.matches[:num_pts_to_visualize, 0]],
            self.X2[self.matches[:num_pts_to_visualize, 1]],
            self.Y2[self.matches[:num_pts_to_visualize, 1]]
        )
        plt.figure(figsize=(10, 5))
        #plt.imshow(c2)
        _save_image('output/vis_lines{}.jpg'.format(self.outputSuffix), c2)


class Matches:
    def __init__(self, matches, confidence, p1, p2, K1, K2):
        self.matches = matches
        self.confidence = confidence
        self.p1 = p1
        self.p2 = p2
        self.K1 = K1
        self.K2 = K2


class SFMRunner:
    def __init__(self, img_path, max_img, pose_estimator: PoseEstimator = None, dist_threshold=5.0, export_suffix=None):
        self.img_path = img_path
        self.pose_estimator = pose_estimator

        self.global_poses = []
        self.global_points_3D = []
        self.global_points_2D = []
        self.frame_indices = []  # Stores the frame index for each 2D point
        self.point_indices = []  # Maps 2D points to their corresponding 3D points
        self.global_K = []

        self.best_matches: Tuple | None = None
        self.processed_pairs = set()

        self.max_img = max_img
        self.dist_threshold = dist_threshold
        self.export_name = export_suffix

        self.perform()

    @staticmethod
    def load(import_name):
        npz = np.load('output/{}.npz'.format(import_name))

        global_points_3D = npz["p3d"].tolist()
        frame_indices = npz["frame_idx"].tolist()
        point_indices = npz["pt_idx"].tolist()

        Visualizer(global_points_3D, frame_indices, point_indices)

    def perform(self):
        self.ransac_max_it = CameraPose.calculate_num_ransac_iterations(0.98, 8, 0.4)
        print("Ransac max iterations {}".format(self.ransac_max_it))

        self.all_matches: list[list[Matches | None]] = [[None for _ in range(self.max_img + 1)] for _ in
                                                        range(self.max_img + 1)]

        # Concurrent stuff, this processing speed is killing me
        self.lock = Lock()
        self.lock2 = Lock()
        tasks = [(i1, i1 + 1) for i1 in range(1, self.max_img)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.corner_detect_and_matching_process, i1, i2) for i1, i2 in tasks]

            # Make sure all threads finish and throwing output out in stdout before moving on
            for future in futures:
                future.result()

        # Initial pose
        self.best_matches = (1, 2)
        print("Found best pair {}".format(self.best_matches))

        # Perform initial ransac to figure out initial structure
        initial_matches = self.all_matches[self.best_matches[0]][self.best_matches[1]]
        R1 = np.eye(3)
        t1 = np.zeros(3)

        cam_pose = CameraPose(initial_matches.p1, initial_matches.p2, initial_matches.K1, initial_matches.K2)
        R2, t2, p1, p2 = cam_pose.ransac_camera_motion(R1, t1, max_iterations=self.ransac_max_it)

        # Triangulate using inliers from ransac
        P1 = CameraPose.calculate_projection_matrix(R1, t1, initial_matches.K1)
        P2 = CameraPose.calculate_projection_matrix(R2, t2, initial_matches.K2)
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

        # Find next frame
        while True:  # For now, loops forever until there's no available next frame
            best_pair = None
            best_next_frame = None
            best_prev_frame = None

            # Take second image as the reference image for the next frame, for consistency
            i = self.best_matches[1]
            j = i + 1

            if i == self.max_img:
                break

            if (i, j) in self.processed_pairs or (self.all_matches[i][j] is None):
                continue

            points_2D_from_prev_frame = p2
            matches = self.all_matches[i][j]
            prev_frame_2d = matches.p1
            next_frame_2d = matches.p2

            result_prev = []
            result_next = []

            # Since 3D points are constructed from at, we need to find all the points that at has in bt, and then
            # find out what that bt points correspond to ct. This is an effort to make sure the 2D points of
            # next frame correspond to already established 3D points
            for p_prime in range(prev_frame_2d.shape[0]):
                dist = Util.compute_euclidean_distance(points_2D_from_prev_frame, prev_frame_2d[p_prime:p_prime + 1])
                mask = np.argmin(dist)

                if dist[mask] < self.dist_threshold:
                    result_prev.append(p3d[mask])
                    result_next.append(next_frame_2d[p_prime])

            best_prev_frame = np.array(result_prev)
            best_next_frame = np.array(result_next)
            best_pair = (i, j)

            print("Pose estimate pair {}".format(best_pair))
            self.processed_pairs.add(best_pair)
            self.processed_pairs.add((best_pair[1], best_pair[0]))

            # Use pose estimation to find current frame R and t
            matches = self.all_matches[best_pair[0]][best_pair[1]]

            K = matches.K2
            pe = self.pose_estimator(np.asarray(best_prev_frame, dtype=np.float32),
                                     np.asarray(best_next_frame, dtype=np.float32),
                                     K=K, ransac_max_it=self.ransac_max_it)

            R3, t3, p_inliers = pe.R, pe.t, pe.inliers
            if R3 is None:
                raise Exception("Cannot determine pose for pair {}".format(best_pair))

            current_frame = self.frame_indices[len(self.frame_indices) - 1] + 1
            if not p_inliers is None:
                self.add_points(best_prev_frame, best_next_frame, current_frame)

            # Set current frame as prev frame to prepare for next iteration
            p1 = matches.p1
            p2 = matches.p2
            P1 = P2
            P2 = CameraPose.calculate_projection_matrix(R3, t3, matches.K2)

            # Triangulate new 3D points and add to the global list
            p3d = np.array([CameraPose.triangulate_point(pts1, pts2, P1, P2) for pts1, pts2 in zip(p1, p2)])
            self.add_points(p3d, p2, current_frame)

            # Keep track of poses and K intrinsics
            R3, _ = cv2.Rodrigues(R3)
            self.global_poses.append((R3, t3))
            self.global_K.append(K)

            # Best match of current frame becomes best match of prev frame
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

        if not self.export_name is None:
            self.save_data()

    def save_data(self):
        np.savez('output/{}.npz'.format(self.export_name), p3d=np.array(self.global_points_3D),
                 frame_idx=np.array(self.frame_indices), pt_idx=np.array(self.point_indices))

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

    def is_new_point(self, p3d, threshold=1e-6):
        if not self.global_points_3D:
            return True

        distances = Util.compute_euclidean_distance(np.array(self.global_points_3D), p3d[np.newaxis])
        return np.min(distances) >= threshold

    def find_existing_point(self, p3d, threshold=1e-6):
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
            'num_interest_points': 2500,
            'ksize': 3,
            'gaussian_size': 7,
            'sigma': 6,
            'alpha': 0.05,
            'feature_width': 18,
            'pyramid_level': 3,
            'pyramid_scale_factor': 1.1
        }

        srunner = FeatureRunner("{}/{}.jpg".format(self.img_path, i1), "{}/{}.jpg".format(self.img_path, i2),
                                feature_extractor_class=ScaleRotInvSIFT, extractor_params=extractor_params,
                                match_threshold=0.85)
        p1, p2 = _convert_matches_to_coords(srunner.matches, srunner.X1, srunner.Y1, srunner.X2, srunner.Y2, 2500)

        if (i1, i2) != (1, 2):
            p1, p2 = CameraPose.find_inliers(p1, p2, max_iterations=self.ransac_max_it)

        with self.lock2:
            self.all_matches[i1][i2] = Matches(srunner.matches, srunner.confidences, p1, p2, K1, K2)
            self.all_matches[i2][i1] = Matches(srunner.matches, srunner.confidences, p2, p1, K2, K1)

            if self.best_matches is None or srunner.matches.shape[0] > \
                    self.all_matches[self.best_matches[0]][self.best_matches[1]].matches.shape[0]:
                self.best_matches = (i1, i2)


###############
### HELPERS ###
###############

def _convert_matches_to_coords(sift_matches, X1, Y1, X2, Y2, num_matches=2500):
    if (sift_matches.shape[0] == 0):
        return np.array([]), np.array([])

    # Extract the first num_matches matches
    match_indices = sift_matches[:num_matches]

    # Extract coordinates using match indices
    pts1 = np.column_stack((X1[match_indices[:, 0]], Y1[match_indices[:, 0]]))
    pts2 = np.column_stack((X2[match_indices[:, 1]], Y2[match_indices[:, 1]]))

    return pts1, pts2


def print_sift_matches(image1, image2, keypoints1, keypoints2, matches, output_path="output/sift_matches.png"):
    plt.figure(figsize=(10, 5))
    matched_img = _show_correspondence_lines(image1, image2,
                                             [kp.pt for kp in keypoints1],
                                             [kp.pt for kp in keypoints2],
                                             matches)
    plt.imshow(matched_img)
    plt.savefig(output_path)
    print(f"Saved SIFT matches to {output_path}")


def _show_interest_points(img, keypoints):
    img = img.copy()
    img = Image.fromarray((img * 255).astype('uint8'))
    draw = ImageDraw.Draw(img)
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=2)
    return np.array(img) / 255


def print_harris_corners(image, keypoints, output_path="output/harris_corners.png"):
    num_pts_to_visualize = min(300, len(keypoints))
    rendered_img = _show_interest_points(image, keypoints[:num_pts_to_visualize])

    plt.figure(figsize=(6, 6))
    plt.imshow(rendered_img, cmap='gray')
    plt.savefig(output_path)


def _rgb2gray(img: np.ndarray) -> np.ndarray:
    """Use the coefficients used in OpenCV, found here:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    Args:
        Numpy array of shape (M,N,3) representing RGB image in HWC format

    Returns:
        Numpy array of shape (M,N) representing grayscale image
    """
    # Grayscale coefficients
    c = [0.299, 0.587, 0.114]
    return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]


def _PIL_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        img: Array representing an image
        size: Tuple representing new desired (width, height)

    Returns:
        img
    """
    img = _numpy_arr_to_PIL_image(img, scale_to_255=True)
    img = img.resize(size)
    img = _PIL_image_to_numpy_arr(img)
    return img


def _PIL_image_to_numpy_arr(img: Image, downscale_by_255: bool = True) -> np.ndarray:
    """
    Args:
        img: PIL Image
        downscale_by_255: whether to divide uint8 values by 255 to normalize
        values to range [0,1]

    Returns:
        img
    """
    img = np.asarray(img)
    img = img.astype(np.float32)
    if downscale_by_255:
        img /= 255
    return img


def _im2single(im: np.ndarray) -> np.ndarray:
    """
    Args:
        img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
        im: float or double array of identical shape and in range [0,1]
    """
    im = im.astype(np.float32) / 255
    return im


def _single2im(im: np.ndarray) -> np.ndarray:
    """
    Args:
        im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
        im: uint8 array of identical shape and in range [0,255]
    """
    im *= 255
    im = im.astype(np.uint8)
    return im


def _numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
    """
    Args:
        img: in [0,1]

    Returns:
        img in [0,255]
    """
    if scale_to_255:
        img *= 255
    return PIL.Image.fromarray(np.uint8(img))


def _load_image(path: str) -> np.ndarray:
    """
    Args:
        path: string representing a file path to an image

    Returns:
        float_img_rgb: float or double array of shape (m,n,c) or (m,n)
           and in range [0,1], representing an RGB image
    """
    img = PIL.Image.open(path)
    img = np.asarray(img, dtype=float)
    float_img_rgb = _im2single(img)
    return float_img_rgb


def _save_image(path: str, im: np.ndarray) -> None:
    """
    Args:
        path: string representing a file path to an image
        img: numpy array
    """
    folder_path = os.path.split(path)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img = copy.deepcopy(im)
    img = _single2im(img)
    pil_img = _numpy_arr_to_PIL_image(img, scale_to_255=False)
    pil_img.save(path)


def _hstack_images(img1, img2):
    """
    Stacks 2 images side-by-side and creates one combined image.

    Args:
    - imgA: A numpy array of shape (M,N,3) representing rgb image
    - imgB: A numpy array of shape (D,E,3) representing rgb image

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """

    # CHANGED
    imgA = np.array(img1)
    imgB = np.array(img2)
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[: imgA.shape[0], : imgA.shape[1], :] = imgA
    newImg[: imgB.shape[0], imgA.shape[1]:, :] = imgB

    # newImg = PIL.Image.fromarray(np.uint8(newImg))
    return newImg


def _show_interest_points(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Visualized interest points on an image with random colors

    Args:
        img: array of shape (M,N,C)
        X: array of shape (k,) containing x-locations of interest points
        Y: array of shape (k,) containing y-locations of interest points

    Returns:
        newImg: A numpy array of shape (M,N,C) showing the original image with
            colored circles at keypoints plotted on top of it
    """
    # CHANGED
    newImg = img.copy()
    newImg = _numpy_arr_to_PIL_image(newImg, True)
    r = 10
    draw = PIL.ImageDraw.Draw(newImg)
    for x, y in zip(X.astype(int), Y.astype(int)):
        cur_color = np.random.rand(3) * 255
        cur_color = (int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=cur_color)

    return _PIL_image_to_numpy_arr(newImg, True)


def _show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
    """
    Visualizes corresponding points between two images by drawing a line
    segment between the two images for each (x1,y1) (x2,y2) pair.

    Args:
        imgA: A numpy array of shape (M,N,3)
        imgB: A numpy array of shape (D,E,3)
        x1: A numpy array of shape (k,) containing x-locations of imgA keypoints
        y1: A numpy array of shape (k,) containing y-locations of imgA keypoints
        x2: A numpy array of shape (j,) containing x-locations of imgB keypoints
        y2: A numpy array of shape (j,) containing y-locations of imgB keypoints
        line_colors: A numpy array of shape (N x 3) with colors of correspondence
            lines (optional)

    Returns:
        newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    newImg = _hstack_images(imgA, imgB)
    newImg = _numpy_arr_to_PIL_image(newImg, True)

    draw = PIL.ImageDraw.Draw(newImg)
    r = 10
    shiftX = imgA.shape[1]
    X1 = X1.astype(int)
    Y1 = Y1.astype(int)
    X2 = X2.astype(int)
    Y2 = Y2.astype(int)

    dot_colors = (np.random.rand(len(X1), 3) * 255).astype(int)
    if line_colors is None:
        line_colors = dot_colors
    else:
        line_colors = (line_colors * 255).astype(int)

    for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors, line_colors):
        # newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        # newImg = cv2.circle(newImg, (x2+shiftX, y2), 5, dot_color, -1)
        # newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 2,
        #                                     cv2.LINE_AA)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=tuple(dot_color))
        draw.ellipse((x2 + shiftX - r, y2 - r, x2 + shiftX + r, y2 + r), fill=tuple(dot_color))
        draw.line((x1, y1, x2 + shiftX, y2), fill=tuple(line_color), width=10)
    return _PIL_image_to_numpy_arr(newImg, True)


def _show_correspondence_circles(imgA, imgB, X1, Y1, X2, Y2):
    """
    Visualizes corresponding points between two images by plotting circles at
    each correspondence location. Corresponding points will have the same
    random color.

    Args:
        imgA: A numpy array of shape (M,N,3)
        imgB: A numpy array of shape (D,E,3)
        x1: A numpy array of shape (k,) containing x-locations of imgA keypoints
        y1: A numpy array of shape (k,) containing y-locations of imgA keypoints
        x2: A numpy array of shape (j,) containing x-locations of imgB keypoints
        y2: A numpy array of shape (j,) containing y-locations of imgB keypoints

    Returns:
        newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    # CHANGED
    newImg = _hstack_images(imgA, imgB)
    newImg = _numpy_arr_to_PIL_image(newImg, True)
    draw = PIL.ImageDraw.Draw(newImg)
    shiftX = imgA.shape[1]
    X1 = X1.astype(int)
    Y1 = Y1.astype(int)
    X2 = X2.astype(int)
    Y2 = Y2.astype(int)
    r = 10
    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        cur_color = np.random.rand(3) * 255
        cur_color = (int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
        green = (0, 1, 0)
        draw.ellipse([x1 - r + 1, y1 - r + 1, x1 + r - 1, y1 + r - 1], fill=cur_color, outline=green)
        draw.ellipse([x2 + shiftX - r + 1, y2 - r + 1, x2 + shiftX + r - 1, y2 + r - 1], fill=cur_color, outline=green)

        # newImg = cv2.circle(newImg, (x1, y1), 10, cur_color, -1, cv2.LINE_AA)
        # newImg = cv2.circle(newImg, (x1, y1), 10, green, 2, cv2.LINE_AA)
        # newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, cur_color, -1,
        #                     cv2.LINE_AA)
        # newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, green, 2, cv2.LINE_AA)

    return _PIL_image_to_numpy_arr(newImg, True)
