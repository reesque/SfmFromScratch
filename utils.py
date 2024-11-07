#!/usr/bin/python3

import os
import copy
import pickle
from typing import Any, Callable, List, Optional, Tuple


import numpy as np
import PIL
# import torch
# import torchvision
from PIL import Image, ImageDraw


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Bring image values to [0,1] range

    Args:
        img: (H,W,C) or (H,W) image
    """
    img -= img.min()
    img /= img.max()
    return img


def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
        function: Python function object
    Returns:
        string that is colored red or green when printed, indicating success
    """
    try:
        function
        return '\x1b[32m"Correct"\x1b[0m'
    except AssertionError:
        return '\x1b[31m"Wrong"\x1b[0m'


def rgb2gray(img: np.ndarray) -> np.ndarray:
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


def PIL_resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        img: Array representing an image
        size: Tuple representing new desired (width, height)

    Returns:
        img
    """
    img = numpy_arr_to_PIL_image(img, scale_to_255=True)
    img = img.resize(size)
    img = PIL_image_to_numpy_arr(img)
    return img


def PIL_image_to_numpy_arr(img: Image, downscale_by_255: bool = True) -> np.ndarray:
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


def im2single(im: np.ndarray) -> np.ndarray:
    """
    Args:
        img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
        im: float or double array of identical shape and in range [0,1]
    """
    im = im.astype(np.float32) / 255
    return im


def single2im(im: np.ndarray) -> np.ndarray:
    """
    Args:
        im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
        im: uint8 array of identical shape and in range [0,255]
    """
    im *= 255
    im = im.astype(np.uint8)
    return im


def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
    """
    Args:
        img: in [0,1]

    Returns:
        img in [0,255]
    """
    if scale_to_255:
        img *= 255
    return PIL.Image.fromarray(np.uint8(img))


def load_image(path: str) -> np.ndarray:
    """
    Args:
        path: string representing a file path to an image

    Returns:
        float_img_rgb: float or double array of shape (m,n,c) or (m,n)
           and in range [0,1], representing an RGB image
    """
    img = PIL.Image.open(path)
    img = np.asarray(img, dtype=float)
    float_img_rgb = im2single(img)
    return float_img_rgb


def save_image(path: str, im: np.ndarray) -> None:
    """
    Args:
        path: string representing a file path to an image
        img: numpy array
    """
    folder_path = os.path.split(path)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img = copy.deepcopy(im)
    img = single2im(img)
    pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
    pil_img.save(path)


def cheat_interest_points(eval_file, scale_factor):
    """
    This function is provided for development and debugging but cannot be used
    in the final hand-in. It 'cheats' by generating interest points from known
    correspondences. It will only work for the 3 image pairs with known
    correspondences.

    Args:
    - eval_file: string representing the file path to the list of known
      correspondences
    - scale_factor: Python float representing the scale needed to map from the
      original image coordinates to the resolution being used for the current
      experiment.

    Returns:
    - x1: A numpy array of shape (k,) containing ground truth x-coordinates of
      imgA correspondence pts
    - y1: A numpy array of shape (k,) containing ground truth y-coordinates of
      imgA correspondence pts
    - x2: A numpy array of shape (k,) containing ground truth x-coordinates of
      imgB correspondence pts
    - y2: A numpy array of shape (k,) containing ground truth y-coordinates of
      imgB correspondence pts
    """
    with open(eval_file, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    return d["x1"] * scale_factor, d["y1"] * scale_factor, d["x2"] * scale_factor, d["y2"] * scale_factor


def hstack_images(img1, img2):
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
    newImg[: imgB.shape[0], imgA.shape[1] :, :] = imgB

    # newImg = PIL.Image.fromarray(np.uint8(newImg))
    return newImg


def show_interest_points(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
    newImg = numpy_arr_to_PIL_image(newImg, True)
    r = 10
    draw = PIL.ImageDraw.Draw(newImg)
    for x, y in zip(X.astype(int), Y.astype(int)):
        cur_color = np.random.rand(3) * 255
        cur_color = (int(cur_color[0]), int(cur_color[1]), int(cur_color[2]))
        # newImg = cv2.circle(newImg, (x, y), 10, cur_color, -1, cv2.LINE_AA
        draw.ellipse([x - r, y - r, x + r, y + r], fill=cur_color)

    return PIL_image_to_numpy_arr(newImg, True)


def show_correspondence_circles(imgA, imgB, X1, Y1, X2, Y2):
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
    newImg = hstack_images(imgA, imgB)
    newImg = numpy_arr_to_PIL_image(newImg, True)
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

    return PIL_image_to_numpy_arr(newImg, True)


def show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
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
    newImg = hstack_images(imgA, imgB)
    newImg = numpy_arr_to_PIL_image(newImg, True)

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
    return PIL_image_to_numpy_arr(newImg, True)


def show_ground_truth_corr(imgA: str, imgB: str, corr_file: str, show_lines: bool = True):
    """
    Show the ground truth correspondeces

    Args:
        imgA: string, representing the filepath to the first image
        imgB: string, representing the filepath to the second image
        corr_file: filepath to pickle (.pkl) file containing the correspondences
        show_lines: boolean, whether to visualize the correspondences as line segments
    """
    imgA = load_image(imgA)
    imgB = load_image(imgB)
    with open(corr_file, "rb") as f:
        d = pickle.load(f)
    if show_lines:
        return show_correspondence_lines(imgA, imgB, d["x1"], d["y1"], d["x2"], d["y2"])
    else:
        # show circles
        return show_correspondence_circles(imgA, imgB, d["x1"], d["y1"], d["x2"], d["y2"])


def load_corr_pkl_file(corr_fpath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Load ground truth correspondences from a pickle (.pkl) file. """
    with open(corr_fpath, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x1 = d["x1"].squeeze()
    y1 = d["y1"].squeeze()
    x2 = d["x2"].squeeze()
    y2 = d["y2"].squeeze()

    return x1, y1, x2, y2


def evaluate_correspondence(
    imgA: np.ndarray,
    imgB: np.ndarray,
    corr_fpath: str,
    scale_factor: float,
    x1_est: np.ndarray,
    y1_est: np.ndarray,
    x2_est: np.ndarray,
    y2_est: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    num_req_matches: int = 100,
) -> Tuple[float, np.ndarray]:
    """
    Function to evaluate estimated correspondences against ground truth.

    The evaluation requires 100 matches to receive full credit
    when num_req_matches=100 because we define accuracy as:

    Let TP = true_pos
    Let FP = false_pos

    Accuracy = (TP)/(TP + FP) * min(num_matches,num_req_matches)/num_req_matches

    Args:
        imgA: A numpy array of shape (M,N,C) representing a first image
        imgB: A numpy array of shape (M,N,C) representing a second image
        corr_fpath: string, representing a filepath to a .pkl file containing
            ground truth correspondences
        scale_factor: scale factor on the size of the images
        x1_est: array of shape (k,) containing estimated x-coordinates of imgA correspondence pts
        y1_est: array of shape (k,) containing estimated y-coordinates of imgA correspondence pts
        x2_est: array of shape (k,) containing estimated x-coordinates of imgB correspondence pts
        y2_est: array of shape (k,) containing estimated y-coordinates of imgB correspondence pts
        confidences: (optional) confidence values in the matches

    Returns:
        acc: accuracy as decimal / ratio (between 0 and 1)
        rendered_img: image with correct matches rendered as green lines, incorrect rendered as red
    """
    if confidences is None:
        confidences = np.random.rand(len(x1_est))
        confidences /= np.max(confidences)

    x1_est = x1_est.squeeze() / scale_factor
    y1_est = y1_est.squeeze() / scale_factor
    x2_est = x2_est.squeeze() / scale_factor
    y2_est = y2_est.squeeze() / scale_factor

    num_matches = x1_est.shape[0]

    x1, y1, x2, y2 = load_corr_pkl_file(corr_fpath)

    good_matches = [False for _ in range(len(x1_est))]
    # array marking which GT pairs are already matched
    matched = [False for _ in range(len(x1))]

    # iterate through estimated pairs in decreasing order of confidence
    priority = np.argsort(-confidences)
    for i in priority:
        # print('Examining ({:4.0f}, {:4.0f}) to ({:4.0f}, {:4.0f})'.format(
        #     x1_est[i], y1_est[i], x2_est[i], y2_est[i]))
        cur_offset = np.asarray([x1_est[i] - x2_est[i], y1_est[i] - y2_est[i]])
        # for each x1_est find nearest ground truth point in x1
        dists = np.linalg.norm(np.vstack((x1_est[i] - x1, y1_est[i] - y1)), axis=0)
        best_matches = np.argsort(dists)

        # find the best match that is not taken yet
        for match_idx in best_matches:
            if not matched[match_idx]:
                break
        else:
            continue

        # A match is good only if
        # (1) An unmatched GT point exists within 150 pixels, and
        # (2) GT correspondence offset is within 25 pixels of estimated
        #     correspondence offset
        gt_offset = np.asarray([x1[match_idx] - x2[match_idx], y1[match_idx] - y2[match_idx]])
        offset_dist = np.linalg.norm(cur_offset - gt_offset)
        if (dists[match_idx] < 150.0) and (offset_dist < 25):
            good_matches[i] = True
            # pass #print('Correct')
        else:
            pass  # print('Incorrect')

    print("You found {}/{} required matches".format(num_matches, num_req_matches))
    accuracy = np.mean(good_matches) * min(num_matches, num_req_matches) * 1.0 / num_req_matches
    print("Accuracy = {:f}".format(accuracy))
    green = np.asarray([0, 1, 0], dtype=float)
    red = np.asarray([1, 0, 0], dtype=float)
    line_colors = np.asarray([green if m else red for m in good_matches])

    rendered_img = show_correspondence_lines(
        imgA,
        imgB,
        x1_est * scale_factor,
        y1_est * scale_factor,
        x2_est * scale_factor,
        y2_est * scale_factor,
        line_colors,
    )

    return accuracy, rendered_img


def compute_feature_distances(
    features1: np.ndarray,
    features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/#broadcasting

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # reshape features1 for broadcasting
    features1_reshaped = np.reshape(features1, (features1.shape[0], 1, features1.shape[1]))

    # subtract the features
    diff = features1_reshaped - features2

    # calculate distances
    dists = np.sqrt(np.sum(diff**2, axis=2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features_ratio_test(
    features1: np.ndarray,
    features2: np.ndarray,
    ratio_thresh: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    matches = []
    confidences = []
    
    # calculate distances between features
    dist = compute_feature_distances(features1, features2)

    for i in range(len(features1)):
        # sort and get indexes
        dist_sorted = np.argsort(dist[i, :])
        
        # get first and second best indexes
        dist_best_idx = dist_sorted[0]
        dist_second_best_idx = dist_sorted[1]
        
        # get distances from indexes
        dist_best = dist[i, dist_best_idx]
        dist_second_best = dist[i, dist_second_best_idx]

        # check if they are within the threshold
        if dist_best / dist_second_best < ratio_thresh:
            matches.append([i, dist_best_idx])
            confidences.append(dist_best)
    
    matches = np.array(matches)
    confidences = np.array(confidences)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences