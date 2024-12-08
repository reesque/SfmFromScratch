import matplotlib

from PoseEstimator import PnPRansac
from Runner import SFMRunner
from sys import platform

from SFM import SensorType

from FeatureExtractor.SuperPoint.SuperPoint import SuperPoint


def main():
    model_name = "bal"
    # extractor_params = {
    #     'num_interest_points': 2000,
    #     'ksize': 3,
    #     'gaussian_size': 7,
    #     'sigma': 6,
    #     'alpha': 0.05,
    #     'feature_width': 18,
    #     'pyramid_level': 3,
    #     'pyramid_scale_factor': 1.1
    # }
    extractor_params = {}

    SFMRunner("test_data/tallneck2_mini", 5, feature_extractor_class=SuperPoint, extractor_params=extractor_params,
              match_threshold=0.85, pose_estimator=PnPRansac, camera_sensor=SensorType.CROP_FRAME, model_name=model_name)
    SFMRunner.load(model_name)
    

if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('TkAgg')

    # Main call
    main()
