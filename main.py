import matplotlib

from PoseEstimator import PnPRansac
from runner import SFMRunner
from sys import platform


def main():
    #SFMRunner("test_data/tallneck2_mini", 3, PnPRansac, export_suffix="exp")
    #SFMRunner.load("best")
    

if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('TkAgg')

    # Main call
    main()
