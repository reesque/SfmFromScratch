from typing import Tuple

import cv2
import numpy as np

from FeatureExtractor import NaiveSIFT

class ScaleRotInvSIFT(NaiveSIFT):
    def __init__(self, image_bw: np.ndarray, extractor_params: dict = {}):
        super().__init__(image_bw, extractor_params)
        
        self._pyramid_level = extractor_params.get('pyramid_level', 4)
        self._pyramid_scale_factor = extractor_params.get('pyramid_scale_factor', 2)
        
        self._build_image_pyramid(self.image, self._pyramid_level, self._pyramid_scale_factor)
        self.compute(image_bw, self.num_interest_points)
    
    def detect_keypoints(self):
        return self._X, self._Y

    def extract_descriptors(self):
        return self._feature_vec
    
    def _compute_dominant_orientation(self, magnitudes, orientations):
        # Create histogram bins for orientations
        hist_bins = np.linspace(-np.pi, np.pi, 37)
        hist, _ = np.histogram(orientations, bins=hist_bins, weights=magnitudes)
        
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        dominant_orientation = bin_centers[np.argmax(hist)]
        return dominant_orientation
    
    def _get_SIFT_descriptors(self, image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """
        This function returns the 128-d SIFT features computed at each of the input
        points
        """
        assert image_bw.ndim == 2, 'Image must be grayscale'

        Ix, Iy = self._compute_image_gradients(image_bw)
        magn = np.sqrt(Ix ** 2 + Iy ** 2)
        orient = np.arctan2(Iy, Ix)

        fvs = []
        for coord in range(X.shape[0]):
            x = X[coord]
            y = Y[coord]

            # Half for lower lim, half for upper lim
            feat_width_half = self._feature_width // 2

            # Define 16x16 feature from the given y:x
            feat_magnitudes = magn[y - feat_width_half + 1: y + feat_width_half + 1,
                              x - feat_width_half + 1: x + feat_width_half + 1]
            feat_orientations = orient[y - feat_width_half + 1: y + feat_width_half + 1,
                                x - feat_width_half + 1: x + feat_width_half + 1]

            # Calculate dominant orientation
            dominant_orientation = self._compute_dominant_orientation(feat_magnitudes, feat_orientations)

            # Adjust orientation to be relative to dominant orientation
            feat_orientations = (feat_orientations - dominant_orientation)

            hist_bin_edges = np.linspace(-np.pi, np.pi, 9)
            wgh = []

            # Loop through 4x4 patches
            for r in range(4):
                for c in range(4):
                    patch_magnitudes = feat_magnitudes[r * 4: (r + 1) * 4, c * 4: (c + 1) * 4]
                    patch_orientations = feat_orientations[r * 4: (r + 1) * 4, c * 4: (c + 1) * 4]

                    hist_data = np.histogram(patch_orientations.flatten(), bins=hist_bin_edges,
                                             weights=patch_magnitudes.flatten())

                    wgh.append(hist_data[0])

            # Stack outputs into np array, then reshape it into desire shape
            wgh = np.vstack(wgh).reshape(128, 1)

            # Normalize and raise to the power of 1/2, RootSIFT
            wgh_norm = np.linalg.norm(wgh)
            if wgh_norm > 0:
                wgh = wgh / wgh_norm
            fvs.append(np.sqrt(wgh))

        return np.squeeze(np.array(fvs))

    def compute(self, image_bw: np.ndarray, k: int):
        scaled_k = int(k / self._pyramid_level)
        self._X, self._Y, self._feature_vec = [], [], []

        for level, scaled_image in enumerate(self._img_pyramid):
            scale = self._pyramid_scale_factor ** level
            scaled_feature_width = int(self._feature_width / scale)

            x, y, _ = self._find_harris_interest_points(scaled_image, scaled_k, scaled_feature_width)
            feat = self._get_SIFT_descriptors(scaled_image, x, y, scaled_feature_width)

            self._X.extend((x * scale).astype(int))
            self._Y.extend((y * scale).astype(int))
            self._feature_vec.extend(feat)

        self._X = np.array(self._X)
        self._Y = np.array(self._Y)
        self._feature_vec = np.array(self._feature_vec)

    def _build_image_pyramid(self, image_bw: np.ndarray, num_levels: int, scale_factor: float):
        """Create an image pyramid with specified number of levels."""
        self._img_pyramid = [image_bw]
        for i in range(1, num_levels):
            self._img_pyramid.append(
                cv2.resize(self._img_pyramid[i - 1], (int(self._img_pyramid[i - 1].shape[0] / (scale_factor ** i)),
                                                      int(self._img_pyramid[i - 1].shape[1] / scale_factor ** i))))

    # def detect_keypoints(self) -> np.ndarray:
    #     """Detects keypoints across image pyramid levels with scale adjustment."""
    #     self._X, self._Y = [], []
    #     scaled_k = int(self._k / self._pyramid_level)

    #     for level, scaled_image in enumerate(self._img_pyramid):
    #         scale = self._pyramid_scale_factor ** level
    #         scaled_feature_width = int(self._feature_width / scale)

    #         x, y, _ = self._find_harris_interest_points(scaled_image, scaled_k, scaled_feature_width)
    #         self._X.extend((x * scale).astype(int))
    #         self._Y.extend((y * scale).astype(int))

    #     self._X = np.array(self._X)
    #     self._Y = np.array(self._Y)
    #     return np.column_stack((self._X, self._Y))

    # def extract_descriptors(self) -> np.ndarray:
    #     """Extracts descriptors for each keypoint across image pyramid levels."""
    #     self._feature_vec = []

    #     for level, scaled_image in enumerate(self._img_pyramid):
    #         scale = self._pyramid_scale_factor ** level
    #         scaled_feature_width = int(self._feature_width / scale)
    #         feat = self._get_SIFT_descriptors(scaled_image, self._X, self._Y, scaled_feature_width)
    #         self._feature_vec.extend(feat)

    #     self._feature_vec = np.array(self._feature_vec)
    #     return self._feature_vec

    # def _compute_dominant_orientation(self, magnitudes, orientations):
    #     hist_bins = np.linspace(-np.pi, np.pi, 37)
    #     hist, _ = np.histogram(orientations, bins=hist_bins, weights=magnitudes)
        
    #     bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    #     dominant_orientation = bin_centers[np.argmax(hist)]
    #     return dominant_orientation
    
    # def _get_SIFT_descriptors(self, image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int):
    #     assert image_bw.ndim == 2, 'Image must be grayscale'

    #     Ix, Iy = self._compute_image_gradients(image_bw)
    #     magn = np.sqrt(Ix ** 2 + Iy ** 2)
    #     orient = np.arctan2(Iy, Ix)

    #     fvs = []
    #     for coord in range(X.shape[0]):
    #         x = X[coord]
    #         y = Y[coord]

    #         feat_width_half = feature_width // 2
    #         feat_magnitudes = magn[y - feat_width_half + 1: y + feat_width_half + 1,
    #                                x - feat_width_half + 1: x + feat_width_half + 1]
    #         feat_orientations = orient[y - feat_width_half + 1: y + feat_width_half + 1,
    #                                    x - feat_width_half + 1: x + feat_width_half + 1]

    #         dominant_orientation = self._compute_dominant_orientation(feat_magnitudes, feat_orientations)
    #         feat_orientations -= dominant_orientation

    #         hist_bin_edges = np.linspace(-np.pi, np.pi, 9)
    #         wgh = []

    #         for r in range(4):
    #             for c in range(4):
    #                 patch_magnitudes = feat_magnitudes[r * 4: (r + 1) * 4, c * 4: (c + 1) * 4]
    #                 patch_orientations = feat_orientations[r * 4: (r + 1) * 4, c * 4: (c + 1) * 4]
    #                 hist_data = np.histogram(patch_orientations.flatten(), bins=hist_bin_edges,
    #                                          weights=patch_magnitudes.flatten())
    #                 wgh.append(hist_data[0])

    #         wgh = np.vstack(wgh).reshape(128, 1)
    #         wgh_norm = np.linalg.norm(wgh)
    #         if wgh_norm > 0:
    #             wgh /= wgh_norm
    #         fvs.append(np.sqrt(wgh))

    #     return np.squeeze(np.array(fvs))

    # def _build_image_pyramid(self, image_bw: np.ndarray, num_levels: int, scale_factor: float):
    #     """Creates an image pyramid for scale-invariant feature detection."""
    #     self._img_pyramid = [image_bw]
    #     for i in range(1, num_levels):
    #         scaled_image = cv2.resize(self._img_pyramid[i - 1], 
    #                                   (int(self._img_pyramid[i - 1].shape[1] / scale_factor),
    #                                    int(self._img_pyramid[i - 1].shape[0] / scale_factor)))
    #         self._img_pyramid.append(scaled_image)