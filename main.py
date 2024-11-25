import matplotlib
from runner import FeatureRunner
from sys import platform
from FeatureExtractor import ScaleRotInvSIFT, NaiveSIFT


def main():
    extractor_params = {
        'num_interest_points': 2500,
        'ksize': 7,
        'gaussian_size': 7,
        'sigma': 5,
        'alpha': 0.05,
        'feature_width': 16,
        'pyramid_level': 3,
        'pyramid_scale_factor': 1.2
    }

    FeatureRunner("test_data/a.jpg", "test_data/b.jpg", 
                  feature_extractor_class=ScaleRotInvSIFT, extractor_params=extractor_params, 
                  print_img=True, print_features=True, print_matches=True)
    


if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('Agg')

    # Main call
    main()
