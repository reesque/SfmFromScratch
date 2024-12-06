import matplotlib.pyplot as plt
from FeatureExtractor import FeatureExtractor
from FeatureMatcher import NNRatioFeatureMatcher

from typing import Tuple
import numpy as np
from utils.utils import _load_image, _PIL_resize, _rgb2gray, _show_interest_points, _show_correspondence_lines, _save_image, ransac_fundamental_matrix

class FeatureRunner:
    def __init__(
        self, im1_path: str, im2_path: str, scale_factor: float = 0.5,
        feature_extractor_class: FeatureExtractor = None, extractor_params: dict = {},
        print_img: bool = False, print_features: bool = False, print_matches: bool = False,
        output_path: str = 'output/vis_lines.jpg'
    ):
        if feature_extractor_class is None:
            raise ValueError("Please provide a feature extractor class")

        self.feature_extractor = feature_extractor_class
        self.image1_name = im1_path.split('/')[:-2]
        self.image2_name = im2_path.split('/')[:-2]
        self._init_images(im1_path, im2_path, scale_factor)
        self._init_feature_extraction(extractor_params)
        self._match_features()
        self._ransac_fundamental_matrix()

        # Optional visualizations
        if print_img:
            self._print_images()
        if print_features:
            self._print_features()
        if print_matches:
            self._print_matches(output_path)

    def _init_images(self, im1_path: str, im2_path: str, scale_factor: float):
        """Loads and preprocesses images."""
        self._image1 = _load_image(im1_path)
        self._image2 = _load_image(im2_path)

        # Rescale images
        self._image1 = _PIL_resize(self._image1, self._scaled_size(self._image1, scale_factor))
        self._image2 = _PIL_resize(self._image2, self._scaled_size(self._image2, scale_factor))

        # Convert to grayscale
        self._image1_bw = _rgb2gray(self._image1)
        self._image2_bw = _rgb2gray(self._image2)

    @staticmethod
    def _scaled_size(image: np.ndarray, scale_factor: float) -> Tuple[int, int]:
        """Calculates new size for the image based on a scale factor."""
        return int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)

    def _init_feature_extraction(self, extractor_params: dict):
        """Initializes feature extraction for both images."""
        self.extractor1 = self.feature_extractor(self._image1_bw, extractor_params)
        self.extractor2 = self.feature_extractor(self._image2_bw, extractor_params)

        self.X1, self.Y1, self.descriptors1 = self._extract_features(self.extractor1)
        self.X2, self.Y2, self.descriptors2 = self._extract_features(self.extractor2)

        print(f'{len(self.X1)} corners in image 1 ({self.image1_name}), {len(self.X2)} corners in image 2 ({self.image2_name})')
        print(f'{len(self.descriptors1)} descriptors in image 1({self.image1_name}), {len(self.descriptors2)} descriptors in image 2({self.image2_name})')

    def _extract_features(self, extractor):
        """Extracts keypoints and descriptors."""
        X, Y = extractor.detect_keypoints()
        descriptors = extractor.extract_descriptors()
        return X, Y, descriptors

    def _ransac_fundamental_matrix(self):
        # check if we have 8 matches to compute the fundamental matrix
        if len(self.matches) < 8:
            print('Not enough matches to compute fundamental matrix')
            return

        im1_indices = self.matches[:, 0]  # Indices in image 1
        im2_indices = self.matches[:, 1]  # Indices in image 2

        # Get matched coordinate pairs for RANSAC
        im1_matches = np.vstack((self.X1[im1_indices], self.Y1[im1_indices])).T  # Nx2 array
        im2_matches = np.vstack((self.X2[im2_indices], self.Y2[im2_indices])).T  # Nx2 array

        # Apply RANSAC
        _, a_inliers, b_inliers = ransac_fundamental_matrix(im1_matches, im2_matches)

        # Map coordinates back to indices
        a_inlier_indices = [
            np.where((self.X1 == x) & (self.Y1 == y))[0][0] for x, y in a_inliers
        ]
        b_inlier_indices = [
            np.where((self.X2 == x) & (self.Y2 == y))[0][0] for x, y in b_inliers
        ]

        # Update matches to only include inliers
        self.matches = np.array([[a, b] for a, b in zip(a_inlier_indices, b_inlier_indices)])
        self.a_inliers = a_inlier_indices  # Indices of inliers in image 1
        self.b_inliers = b_inlier_indices  # Indices of inliers in image 2

        print(f'{len(self.matches)} matches found (after RANSAC)')

    def _match_features(self):
        """Matches features between the two images."""
        self.matcher = NNRatioFeatureMatcher(ratio_threshold=0.7)
        self.matches, self.confidences = self.matcher.match_features_ratio_test(self.descriptors1, self.descriptors2)
        print(f'{len(self.matches)} matches found')

    def _print_images(self):
        """Visualizes the loaded images."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self._image1)
        plt.subplot(1, 2, 2)
        plt.imshow(self._image2)
        plt.savefig('output/visual.png')

    def _print_features(self):
        """Visualizes detected features."""
        if not self.X1.size or not self.X2.size:
            print('No interest points to visualize')
            return

        rendered_img1 = _show_interest_points(self._image1, self.X1, self.Y1)
        rendered_img2 = _show_interest_points(self._image2, self.X2, self.Y2)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rendered_img1, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(rendered_img2, cmap='gray')
        plt.savefig('output/features.png')

    def _print_matches(self, output_path: str = 'output/vis_lines.jpg'):
        """Visualizes matches between features."""
        if len(self.matches) == 0:
            print('No matches to visualize')
            return

        c2 = _show_correspondence_lines(
            self._image1, self._image2,
            self.X1[self.matches[:, 0]], self.Y1[self.matches[:, 0]],
            self.X2[self.matches[:, 1]], self.Y2[self.matches[:, 1]]
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(c2)
        _save_image(output_path, c2)
