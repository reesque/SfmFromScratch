from typing import Tuple

import cv2
import numpy as np


class SIFT:
    def __init__(self):
        self.SOBEL_X_KERNEL = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]
                                        ]).astype(np.float32)

        self.SOBEL_Y_KERNEL = np.array([[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]
                                        ]).astype(np.float32)

    def find_harris_interest_points(self, image_bw: np.ndarray, k: int = 2500, ksize: int = 7,
                                    gaussian_size: int = 7, sigma: float = 5, alpha: float = 0.05,
                                    window_size: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Harris Corner detector

        Args:
            image_bw: array of shape (M,N) containing the grayscale image
            k: maximum number of interest points to retrieve
            ksize: kernel size of the max-pooling operator
            gaussian_size: size of 2d Gaussian filter
            sigma: standard deviation of gaussian filter
            alpha: scalar term in Harris response score
            window_size: sift window size

        Returns:
            x: array of shape (p,) containing x-coordinates of interest points
            y: array of shape (p,) containing y-coordinates of interest points
            c: array of dim (p,) containing the strength(confidence) of each
                interest point where p <= k.
        """

        # Second moment
        Ix, Iy = self._compute_image_gradients(image_bw)
        Ix_square = Ix ** 2
        Iy_square = Iy ** 2
        IxIy = Ix * Iy

        gaussian_kernel = self._generate_gaussian_kernel(gaussian_size, sigma)

        S_xx = cv2.filter2D(Ix_square, ddepth=-1, kernel=gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
        S_xy = cv2.filter2D(IxIy, ddepth=-1, kernel=gaussian_kernel, borderType=cv2.BORDER_CONSTANT)
        S_yy = cv2.filter2D(Iy_square, ddepth=-1, kernel=gaussian_kernel, borderType=cv2.BORDER_CONSTANT)

        det_M = (S_xx * S_yy) - (S_xy ** 2)
        trace_M = S_xx + S_yy

        R_map = det_M - alpha * (trace_M ** 2)

        # Max pooling
        Rh, Rw = R_map.shape

        # Since the pixel is in the middle of the kernel, we need to distribute half of the pool to each direction
        ksize_half = ksize // 2

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

        half_window = window_size // 2
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

    def get_SIFT_descriptors(self, image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray,
                             feature_width: int = 16) -> np.ndarray:
        """
        This function returns the 128-d SIFT features computed at each of the input
        points

        Args:
            image: A numpy array of shape (m,n), the image
            X: A numpy array of shape (k,), the x-coordinates of interest points
            Y: A numpy array of shape (k,), the y-coordinates of interest points
            feature_width: integer representing the local feature width in pixels.
                You can assume that feature_width will be a multiple of 4 (i.e.,
                every cell of your local SIFT-like feature will have an integer
                width and height). This is the initial window size we examine
                around each keypoint.
        Returns:
            fvs: A numpy array of shape (k, feat_dim) representing all feature
                vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
                standard SIFT). These are the computed features.
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
            feat_width_half = feature_width // 2

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
            for y in range(4):
                for x in range(4):
                    patch_magnitudes = feat_magnitudes[y * 4: (y + 1) * 4, x * 4: (x + 1) * 4]
                    patch_orientations = feat_orientations[y * 4: (y + 1) * 4, x * 4: (x + 1) * 4]

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

        fvs = np.squeeze(np.array(fvs))

        return fvs

    def _build_image_pyramid(self, image_bw: np.ndarray, num_levels: int = 4, scale_factor: float = 2):
        """Create an image pyramid with specified number of levels."""
        pyramid = [image_bw]
        for i in range(1, num_levels):
            pyramid.append(cv2.resize(image_bw, (pyramid[0][0] * (scale_factor**i), pyramid[0][1] * (scale_factor**i))))

        return pyramid

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

        Args:
            image_bw: A numpy array of shape (M,N) containing the grayscale image

        Returns:
            Ix: Array of shape (M,N) representing partial derivatives of image
                w.r.t. x-direction
            Iy: Array of shape (M,N) representing partial derivative of image
                w.r.t. y-direction
        """

        return (cv2.filter2D(image_bw, ddepth=-1, kernel=self.SOBEL_X_KERNEL, borderType=cv2.BORDER_CONSTANT),
                cv2.filter2D(image_bw, ddepth=-1, kernel=self.SOBEL_Y_KERNEL, borderType=cv2.BORDER_CONSTANT))
