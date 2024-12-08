import matplotlib

from runner import FeatureRunner, SFMRunner
from sys import platform


def main():
    SFMRunner("test_data/tallneck2_mini")
    

if __name__ == "__main__":
    # X11 compatibility
    if platform == "linux" or platform == "linux2":
        matplotlib.use('TkAgg')

    # Main call
    main()
