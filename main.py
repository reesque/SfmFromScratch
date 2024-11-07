import matplotlib
from runner import SIFTRunner
from sys import platform
from SIFT import ScaleRotInvSIFT


def main():
    SIFTRunner("test_data/a.jpg", "test_data/b.jpg", sift_model=ScaleRotInvSIFT, print_img=True, print_harris=True, print_sift=True)


if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('Agg')

    # Main call
    main()
