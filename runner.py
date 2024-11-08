import matplotlib.pyplot as plt

import os
import copy
from typing import Tuple

import numpy as np
import PIL
from PIL import Image, ImageDraw
from SIFT import NaiveSIFT


class SIFTRunner:
    def __init__(self, im1_path: str, im2_path: str, sift_model=NaiveSIFT, num_interest_points: int = 2500,
                 print_img: bool = False,
                 print_harris: bool = False, print_sift: bool = False):
        self.sift_model = sift_model

        self._image1 = _load_image(im1_path)
        self._image2 = _load_image(im2_path)

        scale_factor = 0.5
        self._image1 = _PIL_resize(self._image1,
                                   (int(self._image1.shape[1] * scale_factor),
                                    int(self._image1.shape[0] * scale_factor)))
        self._image2 = _PIL_resize(self._image2,
                                   (int(self._image2.shape[1] * scale_factor),
                                    int(self._image2.shape[0] * scale_factor)))

        self._image1_bw = _rgb2gray(self._image1)
        self._image2_bw = _rgb2gray(self._image2)

        self._sift1 = sift_model(self._image1_bw, k=num_interest_points)
        self._sift2 = sift_model(self._image2_bw, k=num_interest_points)
        self._harris_corner_det()
        self._sift()

        # Optional printing
        if print_img:
            self.print_image()
        if print_harris:
            self.print_harris()
        if print_sift:
            self.print_sift()

    def print_image(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self._image1)
        plt.subplot(1, 2, 2)
        plt.imshow(self._image2)
        plt.savefig('output/visual.png')

    def print_harris(self):
        num_pts_to_visualize = 300
        # Visualize the interest points
        rendered_img1 = _show_interest_points(self._image1, self.X1[:num_pts_to_visualize],
                                              self.Y1[:num_pts_to_visualize])
        rendered_img2 = _show_interest_points(self._image2, self.X2[:num_pts_to_visualize],
                                              self.Y2[:num_pts_to_visualize])
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1);
        plt.imshow(rendered_img1, cmap='gray')
        plt.subplot(1, 2, 2);
        plt.imshow(rendered_img2, cmap='gray')
        print(f'{len(self.X1)} corners in image 1, {len(self.X2)} corners in image 2')
        plt.savefig('output/harris.png')

    def print_sift(self):
        # num_pts_to_visualize = len(matches)
        num_pts_to_visualize = 2500
        c2 = _show_correspondence_lines(
            self._image1,
            self._image2,
            self.X1[self.sift_matches[:num_pts_to_visualize, 0]],
            self.Y1[self.sift_matches[:num_pts_to_visualize, 0]],
            self.X2[self.sift_matches[:num_pts_to_visualize, 1]],
            self.Y2[self.sift_matches[:num_pts_to_visualize, 1]]
        )
        plt.figure(figsize=(10, 5))
        plt.imshow(c2)
        _save_image('output/vis_lines.jpg', c2)

    def _harris_corner_det(self):
        self.X1, self.Y1 = self._sift1.harris_map()
        self.X2, self.Y2 = self._sift2.harris_map()

    def _sift(self):
        self.image1_features = self._sift1.descriptor()
        self.image2_features = self._sift2.descriptor()

        self.sift_matches, self.sift_confidences = self.sift_model.match_features_ratio_test(self.image1_features,
                                                                                             self.image2_features)
        print('{:d} matches from {:d} corners'.format(len(self.sift_matches), len(self.X1)))


###############
### HELPERS ###
###############

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