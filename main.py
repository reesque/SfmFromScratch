import matplotlib

from FeatureExtractor.SIFT.ScaleRotInvSIFT import ScaleRotInvSIFT
from runner import FeatureRunner, SFMRunner
from sys import platform


def main():
    SFMRunner()
    

if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('TkAgg')

    # Main call
    main()
