import matplotlib
import numpy as np

from PoseEstimator import PnPRansac
from Runner import SFMRunner
from sys import platform

from SFM import SensorType


def main():
    model_name = "bal"
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

    SFMRunner("test_data/tallneck2_mini", 5, extractor_params, match_threshold=0.85,
              pose_estimator=PnPRansac, model_name=model_name, camera_sensor=SensorType.CROP_FRAME)
    SFMRunner.load(model_name)
    

if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('TkAgg')

    # Main call
    main()
