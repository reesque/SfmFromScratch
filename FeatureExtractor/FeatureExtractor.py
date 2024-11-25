from abc import ABC, abstractmethod
import numpy as np

class FeatureExtractor(ABC):
    """Interface for feature extractor classes (on images)."""
    
    def __init__(self, image: np.ndarray, extractor_params: dict = {}):
        self.image = image
        self.num_interest_points = extractor_params.get('num_interest_points', 2500)
    
    @abstractmethod
    def detect_keypoints(self) -> np.ndarray:
        """Detects keypoints in the image and returns their coordinates."""
        pass
    
    @abstractmethod
    def extract_descriptors(self) -> np.ndarray:
        """Extracts descriptors for the detected keypoints."""
        pass