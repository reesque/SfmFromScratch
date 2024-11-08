from typing import Tuple

import cv2
import numpy as np

from FeatureExtractor import iFeatureExtractor


class NaiveSIFT(iFeatureExtractor):
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

        self.image = image_bw
        self._ksize = ksize
        self._gaussian_size = gaussian_size
        self._sigma = sigma
        self._alpha = alpha
        self._feature_width = feature_width
        self._k = k

    def detect_keypoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Detect interest points using Harris corner detection."""
        self._X, self._Y, self.confidences = self._find_harris_interest_points(self.image, self._k, self._feature_width)
        return self._X, self._Y
    
    def extract_descriptors(self) -> np.ndarray:
        """Extract SIFT descriptors at detected keypoints."""
        if not hasattr(self, '_X') or not hasattr(self, '_Y'):
            raise RuntimeError("Keypoints not detected. Call detect_keypoints() before extract_descriptors().")
        self.descriptors = self._get_SIFT_descriptors(self.image, self._X, self._Y, self._feature_width)
        return self.descriptors

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

            # Calculate histogram
            # 8 bins + 1 since exclusive limit
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