from typing import Tuple

import cv2
import numpy as np


class NaiveSIFT:
    def __init__(self, image_bw: np.ndarray, k: int = 2500, ksize: int = 7,
                 gaussian_size: int = 7, sigma: float = 5, alpha: float = 0.05,
                 feature_width: int = 16):
        """
        Initialize a SIFT descriptor

        Args:
            k: maximum number of interest points to retrieve
            ksize: kernel size of the max-pooling operator
            gaussian_size: size of 2d Gaussian filter
            sigma: standard deviation of gaussian filter
            alpha: scalar term in Harris response score
            feature_width: sift window size
        """
        self.SOBEL_X_KERNEL = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]
                                        ]).astype(np.float32)

        self.SOBEL_Y_KERNEL = np.array([[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]
                                        ]).astype(np.float32)

        self._ksize = ksize
        self._gaussian_size = gaussian_size
        self._sigma = sigma
        self._alpha = alpha
        self._feature_width = feature_width

        self.compute(image_bw, k)

    def compute(self, image_bw: np.ndarray, k: int):
        self._X, self._Y, _ = self._find_harris_interest_points(image_bw, k, self._feature_width)
        self._feature_vec = self._get_SIFT_descriptors(image_bw, self._X, self._Y, self._feature_width)

    def harris_map(self):
        return self._X, self._Y

    def descriptor(self):
        return self._feature_vec

    @staticmethod
    def match_features_ratio_test(features1: np.ndarray, features2: np.ndarray,
                                  ratio_thresh: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """ Nearest-neighbor distance ratio feature matching.

        This function does not need to be symmetric (e.g. it can produce different
        numbers of matches depending on the order of the arguments).

        Args:
            features1: A numpy array of shape (n1,feat_dim) representing one set of
                features, where feat_dim denotes the feature dimensionality
            features2: A numpy array of shape (n2,feat_dim) representing a second
                set of features (n1 not necessarily equal to n2)
            ratio_thresh: ratio for feature comparison

        Returns:
            matches: A numpy array of shape (k,2), where k is the number of matches.
                The first column is an index in features1, and the second column is
                an index in features2
            confidences: A numpy array of shape (k,) with the real valued confidence
                for every match

        'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
        """

        a = features1[:, np.newaxis]
        b = features2[np.newaxis, :]
        # Using Euclidean dist, sqrt(sum((a - b) ** 2))
        dists = np.sqrt(np.sum((a - b) ** 2, axis=2))

        matches = []
        confidences = []

        for d in range(dists.shape[0]):
            # Sort dists
            sorted_dists_idx = np.argsort(dists[d])

            closest_idx = sorted_dists_idx[0]
            second_closest_idx = sorted_dists_idx[1]
            # Divide the closest dist with the second closest dist
            if dists[d, second_closest_idx] > 0:
                nndr = dists[d, closest_idx] / dists[d, second_closest_idx]

                if nndr <= ratio_thresh:
                    matches.append([d, closest_idx])
                    confidences.append(nndr)

        matches = np.array(matches)
        confidences = np.array(confidences)

        return matches, confidences

    def _find_harris_interest_points(self, image_bw: np.ndarray, k: int, feature_width: int):
        """
        Harris Corner detector
        """

        # Second moment
        Ix, Iy = self._compute_image_gradients(image_bw)
        Ix_square = Ix ** 2
        Iy_square = Iy ** 2
        IxIy = Ix * Iy

        gaussian_kernel = self._generate_gaussian_kernel(self._gaussian_size, self._sigma)

        S_xx = cv2.filter2D(Ix_square, ddepth=-1, kernel=gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
        S_xy = cv2.filter2D(IxIy, ddepth=-1, kernel=gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
        S_yy = cv2.filter2D(Iy_square, ddepth=-1, kernel=gaussian_kernel, borderType=cv2.BORDER_CONSTANT)

        det_M = (S_xx * S_yy) - (S_xy ** 2)
        trace_M = S_xx + S_yy

        R_map = det_M - self._alpha * (trace_M ** 2)

        # Max pooling
        Rh, Rw = R_map.shape

        # Since the pixel is in the middle of the kernel, we need to distribute half of the pool to each direction
        ksize_half = self._ksize // 2

        R_maxpool = np.zeros(R_map.shape)

        # Since upper limit is exclusive, all operations with upper limit needs +1
        for r in range(Rh):
            for c in range(Rw):
                R_maxpool[r, c] = np.max(R_map[max(0, r - ksize_half): min(Rh, r + ksize_half + 1),
                                         max(0, c - ksize_half): min(Rw, c + ksize_half + 1)])

        # remove low value scores
        median_R = np.median(R_map)
        R_maxpool[R_map < median_R] = 0

        # Find all the possible results where the R value is the local max
        matching_filter = (R_map == R_maxpool)
        y, x = np.where(matching_filter)
        confidences = R_map[y, x]

        # Sort the result by desc and limit to k records
        sort_filter = np.argsort(confidences)[::-1][:k]
        y = y[sort_filter]
        x = x[sort_filter]
        c = confidences[sort_filter]

        half_window = feature_width // 2
        img_h, img_w = image_bw.shape

        edge_filter = (y >= half_window) & (y < img_h - half_window) & (x >= half_window) & (x < img_w - half_window)

        x = x[edge_filter]
        y = y[edge_filter]
        c = c[edge_filter]

        # Sort interest points by confidences
        indices = np.argsort(c)[::-1][:k]
        x = x[indices]
        y = y[indices]
        c = c[indices]

        return x, y, c
    
    def compute_dominant_orientation(self, magnitudes, orientations):
        # Create histogram bins for orientations
        hist_bins = np.linspace(-np.pi, np.pi, 37)
        hist, _ = np.histogram(orientations, bins=hist_bins, weights=magnitudes)
        
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        dominant_orientation = bin_centers[np.argmax(hist)]
        return dominant_orientation

    def _get_SIFT_descriptors(self, image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int):
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
            dominant_orientation = self.compute_dominant_orientation(feat_magnitudes, feat_orientations)

            # Adjust orientation to be relative to dominant orientation
            feat_orientations = (feat_orientations - dominant_orientation) % (2 * np.pi)

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

    def _generate_gaussian_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        """Create a numpy matrix representing a 2d Gaussian kernel

        Args:
            ksize: dimension of square kernel
            sigma: standard deviation of Gaussian

        Returns:
            kernel: Array of shape (ksize,ksize) representing a 2d Gaussian kernel
        """

        # Since the kernel will be ksize x ksize, one column or row will have the range (-mean, mean)
        mean = ksize // 2
        axis = np.linspace(-mean, mean, ksize)

        # Apply spacially weighted average function with sigma
        x_square = axis[:, np.newaxis] ** 2
        y_square = axis[np.newaxis, :] ** 2

        kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x_square + y_square) / (2 * sigma ** 2))

        # Normalize it
        kernel = kernel / np.sum(kernel)

        return kernel

    def _compute_image_gradients(self, image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use convolution with Sobel filters to compute the image gradient at each
        pixel.

        Returns:
            Ix: Array of shape (M,N) representing partial derivatives of image
                w.r.t. x-direction
            Iy: Array of shape (M,N) representing partial derivative of image
                w.r.t. y-direction
        """

        return (cv2.filter2D(image_bw, ddepth=-1, kernel=self.SOBEL_X_KERNEL, borderType=cv2.BORDER_CONSTANT),
                cv2.filter2D(image_bw, ddepth=-1, kernel=self.SOBEL_Y_KERNEL, borderType=cv2.BORDER_CONSTANT))


class ScaleRotInvSIFT(NaiveSIFT):
    def __init__(self, image_bw: np.ndarray, k: int = 2500, ksize: int = 7,
                 gaussian_size: int = 7, sigma: float = 5, alpha: float = 0.05,
                 feature_width: int = 16, pyramid_level: int = 4, pyramid_scale_factor: float = 2):
        self._build_image_pyramid(image_bw, pyramid_level, pyramid_scale_factor)
        self._pyramid_scale_factor = pyramid_scale_factor
        self._pyramid_level = pyramid_level
        super().__init__(image_bw, k, ksize, gaussian_size, sigma, alpha, feature_width)

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
                                                      int(self._img_pyramid[i - 1].shape[1] / (scale_factor ** i)))))
