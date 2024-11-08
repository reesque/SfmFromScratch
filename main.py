import matplotlib
from runner import FeatureRunner
from sys import platform
from FeatureExtractor import ScaleRotInvSIFT, NaiveSIFT


def main():
    FeatureRunner("test_data/a.jpg", "test_data/b.jpg", feature_extractor_class=NaiveSIFT, 
                  print_img=True, print_features=True, print_matches=True)
    #featureRunner.print_features()
    #featureRunner.print_image()


if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('Agg')

    # Main call
    main()
