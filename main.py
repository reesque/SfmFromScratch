import matplotlib
from PIL import Image
from PIL.ExifTags import TAGS

from SFM import SensorType
#from runner import FeatureRunner, SFMRunner
from feature_runner import FeatureRunner
from SFM_runner import SFMRunner, OpenCVSFMRunner
from sys import platform
from FeatureExtractor import ScaleRotInvSIFT, NaiveSIFT
import numpy as np

from CV_SFM import OpenCVSFM

def construct_K(imagePath, sensorType: SensorType):
    image = Image.open(imagePath)
    width, height = image.size
    exif_data = image._getexif()

    # Decode EXIF data and extract focal length
    focal_length = None
    if exif_data:
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                # If focal length is usually a tuple (numerator, denominator)
                if isinstance(value, tuple):
                    focal_length = value[0] / value[1]  # Convert to a single value
                else:
                    focal_length = value
                break
    else:
        print("No EXIF data. Cannot work with this image")
        raise Exception("No EXIF data. Cannot work with this image")

    if focal_length is None:
        print("No focal length data. Cannot work with this image")
        raise Exception("No focal length data. Cannot work with this image")

    sensor_height = 0.0
    sensor_width = 0.0

    if sensorType is SensorType.MEDIUM_FORMAT:
        sensor_width = 53.0
        sensor_height = 40.20
    elif sensorType is SensorType.FULL_FRAME:
        sensor_width = 35.0
        sensor_height = 24.0
    elif sensorType is SensorType.CROP_FRAME:
        sensor_width = 23.6
        sensor_height = 15.60
    elif sensorType is SensorType.MICRO_FOUR_THIRD:
        sensor_width = 17.0
        sensor_height = 13.0
    elif sensorType is SensorType.ONE_INCH:
        sensor_width = 12.80
        sensor_height = 9.60
    elif sensorType is SensorType.SMARTPHONE:
        sensor_width = 6.17
        sensor_height = 4.55

    fx = focal_length * width / sensor_width
    fy = focal_length * height / sensor_height
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    return K

def main():
    # extractor_params = {
    #     'num_interest_points': 3000,
    #     'ksize': 3,
    #     'gaussian_size': 8,
    #     'sigma': 1,
    #     'alpha': 0.02,
    #     'feature_width': 32,
    #     'pyramid_level': 8,
    #     'pyramid_scale_factor': 1.2
    # }

    extractor_params = {
        'num_interest_points': 3000,
        'ksize': 2,
        'gaussian_size': 6,
        'sigma': 1,
        'alpha': 0.02,
        'feature_width': 32,
        'pyramid_level': 8,
        'pyramid_scale_factor': 1.2
    }

    # for i in range(1, 12):
    #     FeatureRunner(f"test_data/tallneck_mini/{i}.jpg", f"test_data/tallneck_mini/{i+1}.jpg",
    #                   feature_extractor_class=ScaleRotInvSIFT, extractor_params=extractor_params,
    #                   print_img=True, print_features=True, print_matches=True, output_path=f"output/{i}-{i+1}-lines.jpg")


    # K = construct_K("test_data/tallneck_mini/1.jpg", SensorType.FULL_FRAME)
    # image_paths = ["test_data/tallneck_mini/1.jpg", "test_data/tallneck_mini/2.jpg", "test_data/tallneck_mini/3.jpg",]
    #                # "test_data/tallneck_mini/4.jpg", "test_data/tallneck_mini/5.jpg", "test_data/tallneck_mini/6.jpg",]
    #                # "test_data/tallneck_mini/7.jpg", "test_data/tallneck_mini/8.jpg", "test_data/tallneck_mini/9.jpg",
    #                # "test_data/tallneck_mini/10.jpg", "test_data/tallneck_mini/11.jpg", "test_data/tallneck_mini/12.jpg",]
    # sfm_runner = OpenCVSFMRunner(image_paths, K)
    # sfm_runner.run()

    # FeatureRunner("test_data/a.jpg", "test_data/b.jpg",
    #               feature_extractor_class=ScaleRotInvSIFT, extractor_params=extractor_params,
    #               print_img=True, print_features=True, print_matches=True)

    sfm_runner = SFMRunner()

    # print("Initial Reprojection Error:")
    # print(compute_reprojection_error(
    #     sfm_runner.global_points_3d,
    #     sfm_runner.global_camera_poses,
    #     sfm_runner.points_2d,
    #     sfm_runner.camera_indices,
    #     sfm_runner.point_indices,
    #     K
    # ))

    # K = np.eye(3)

    # # Pass K to bundle adjustment
    # sfm_runner.bundle_adjustment_opencv(K)

    # print("Reprojection Error After Bundle Adjustment:")
    # print(compute_reprojection_error(
    #     sfm_runner.global_points_3d,
    #     sfm_runner.global_camera_poses,
    #     points_2d,
    #     camera_indices,
    #     point_indices,
    #     K
    # ))


if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('Agg')

    # Main call
    main()
