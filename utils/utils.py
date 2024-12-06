import os
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
import copy

def _plot_3d_points(points_3d):
    points_3d = [point for point in points_3d if point is not None]
    points_3d = np.array(points_3d)
    
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', s=1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def _convert_matches_to_coords(sift_matches, X1, Y1, X2, Y2, num_matches=2500):
    # Extract the first num_matches matches
    match_indices = sift_matches[:num_matches]

    if len(match_indices) == 0:
        return None, None

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


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    # convert to homogeneous coordinates if not already
    if points.shape[1] == 2:
        points = np.hstack([points, np.ones((points.shape[0], 1))])

    points = points.astype(np.float64)

    u = points[:, 0]
    v = points[:, 1]

    # get means of u and v
    c_u = np.mean(u)
    c_v = np.mean(v)

    # subtract the means
    u_centered = u - c_u
    v_centered = v - c_v

    # Instructions say to use mean squared distance, but this did not pass the test.
    # Instead, mean euclidean distance passed, so I implemented this and kept the
    # mean squared distance code for reference.

    # Mean Euclidean Distance (MED) (passed test)
    # --------------------------------------------------------
    med = np.mean(np.sqrt(u_centered**2 + v_centered**2))

    # get scale (mean euclidean distance approach)
    scale = np.sqrt(2) / med

    # Mean Squared Distance (MSD) (did not pass test so it's commented out)
    # --------------------------------------------------------
    # std_u = np.std(u_centered)
    # std_v = np.std(v_centered)
    # msd = std_u**2 + std_v**2
    # scale = np.sqrt(2) / np.sqrt(msd)

    # define T matrices
    m_1 = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    m_2 = np.array([
        [1, 0, -c_u],
        [0, 1, -c_v],
        [0, 0, 1]
    ])

    # define T
    T = np.dot(m_1, m_2)

    # normalize points
    points_normalized = np.dot(T, points.T).T

    return points_normalized, T


def unnormalize_F(
        F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return T_b.T @ F_norm @ T_a


def estimate_fundamental_matrix(
        points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    N = points_a.shape[0]

    # normalize points
    points_a_norm, T_a = normalize_points(points_a)
    points_b_norm, T_b = normalize_points(points_b)

    U = []
    # build U matrix
    for i in range(N):
        x = points_a_norm[i][0]
        y = points_a_norm[i][1]
        x_p = points_b_norm[i][0]
        y_p = points_b_norm[i][1]
        U.append([x_p * x, x_p * y, x_p, y_p * x, y_p * y, y_p, x, y, 1])
    U = np.array(U)

    # find solution with SVD
    _, _, Vt = np.linalg.svd(U)

    # get initial estimate
    F = Vt[-1].reshape(3, 3)

    # run svd on initial estimate
    U_svd, S, Vt = np.linalg.svd(F)

    # set smallest singular value to 0
    S[2] = 0

    # reconstruct F
    F = np.dot(U_svd, np.dot(np.diag(S), Vt))

    # unnormalize F
    F = unnormalize_F(F, T_a, T_b)

    return F


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
    # calculate the number of iterations
    # equation derived from 1 - (1 - r^S)^N = prob_success
    num_samples = np.log(1 - prob_success) / np.log((1 - ind_prob_correct**sample_size))

    return int(num_samples)


def ransac_fundamental_matrix(
        matches_a: np.ndarray, matches_b: np.ndarray) -> np.ndarray:

    best_F, inliers_a, inliers_b = None, None, None

    M = matches_a.shape[0]

    # initialize threshold
    error_threshold = 12

    # initialize values to determine num_trials
    prob_success = 0.98
    sample_size = 8
    ind_prob_correct = 0.4

    num_trials = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)

    bestCount = -1

    for i in range(num_trials):
        # get subset of indices
        selected_indices = np.random.choice(M, size=sample_size, replace=False)
        a_subset = matches_a[selected_indices]
        b_subset = matches_b[selected_indices]

        # get fundamental matrix
        F = estimate_fundamental_matrix(a_subset, b_subset)

        # calculate error by taking geometric distances of a keypoint to its epipolar line

        # calculate geometric distances from points in image A to epipolar lines in image  B
        l_b = (F @ np.vstack((matches_a[:, 0], matches_a[:, 1], np.ones(M)))).T
        b_a, b_b, b_c = l_b[:, 0], l_b[:, 1], l_b[:, 2]
        xb_to_lb_distances = np.abs(b_a * matches_b[:, 0] + b_b * matches_b[:, 1] + b_c) / np.sqrt(b_a ** 2 + b_b ** 2)

        # calculate geometric distances from points in image B to epipolar lines in image A
        l_a = (F.T @ np.vstack((matches_b[:, 0], matches_b[:, 1], np.ones(M)))).T
        a_a, a_b, a_c = l_a[:, 0], l_a[:, 1], l_a[:, 2]
        xa_to_la_distances = np.abs(a_a * matches_a[:, 0] + a_b * matches_a[:, 1] + a_c) / np.sqrt(a_a ** 2 + a_b ** 2)

        # compute total error
        error = xb_to_lb_distances + xa_to_la_distances

        # get current number of inliers
        curr_inliers_a = matches_a[error <= error_threshold]
        curr_inliers_b = matches_b[error <= error_threshold]

        # check if current F and inliers are best and keep track of them if they are
        if len(curr_inliers_a) > bestCount:
            best_F = F
            inliers_a = curr_inliers_a
            inliers_b = curr_inliers_b
            bestCount = len(inliers_a)

    return best_F, inliers_a, inliers_b