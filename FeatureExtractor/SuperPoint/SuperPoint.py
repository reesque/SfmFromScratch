import PIL
import numpy as np

from FeatureExtractor.SuperPoint.SuperPointModel import SuperPointFrontend
from FeatureExtractor import FeatureExtractor

def _im2single(im: np.ndarray) -> np.ndarray:
    """
    Args:
        img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
        im: float or double array of identical shape and in range [0,1]
    """
    im = im.astype(np.float32) / 255
    return im

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

class SuperPoint(FeatureExtractor):
    def __init__(self, image_bw: np.ndarray, image: np.ndarray, extractor_params=None):
        super().__init__(image, extractor_params)
        if extractor_params is None:
            extractor_params = {}
        self.superpoint = SuperPointFrontend(weights_path='FeatureExtractor/SuperPoint/superpoint_v1.pth', nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
        self._X = []
        self._Y = []
        self._feature_vec = []

        pts, desc, _ = self.superpoint.run(image_bw)
        keypoints = pts[:2, :].T
        self._X = keypoints[:, 0]
        self._Y = keypoints[:, 1]
        self._feature_vec = desc.T

    def detect_keypoints(self):
        return self._X, self._Y

    def extract_descriptors(self):
        return self._feature_vec

