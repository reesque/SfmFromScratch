from abc import ABC, abstractmethod
import numpy as np

class iFeatureExtractor(ABC):
    """Interface for feature extractor classes."""
    
    def __init__(self, image: np.ndarray, num_interest_points: int = 2500):
        self.image = image
        self.num_interest_points = num_interest_points
    
    @abstractmethod
    def detect_keypoints(self) -> np.ndarray:
        """Detects keypoints in the image and returns their coordinates."""
        pass
    
    @abstractmethod
    def extract_descriptors(self) -> np.ndarray:
        """Extracts descriptors for the detected keypoints."""
        pass