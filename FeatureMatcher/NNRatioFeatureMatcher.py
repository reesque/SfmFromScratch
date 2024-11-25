import numpy as np
from typing import Tuple

class NNRatioFeatureMatcher:
    def __init__(self, ratio_threshold=0.8):
        self.ratio_threshold = ratio_threshold

    def match_features_ratio_test(self, features1: np.ndarray, features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Nearest-neighbor distance ratio feature matching.

        This function does not need to be symmetric (e.g. it can produce different
        numbers of matches depending on the order of the arguments).

        Args:
            features1: A numpy array of shape (n1,feat_dim) representing one set of
                features, where feat_dim denotes the feature dimensionality
            features2: A numpy array of shape (n2,feat_dim) representing a second
                set of features (n1 not necessarily equal to n2)
            ratio_thresh: ratio for feature comparison

        Returns:
            matches: A numpy array of shape (k,2), where k is the number of matches.
                The first column is an index in features1, and the second column is
                an index in features2
            confidences: A numpy array of shape (k,) with the real valued confidence
                for every match

        'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
        """

        a = features1[:, np.newaxis]
        b = features2[np.newaxis, :]
        # Using Euclidean dist, sqrt(sum((a - b) ** 2))
        dists = np.sqrt(np.sum((a - b) ** 2, axis=2))

        matches = []
        confidences = []

        for d in range(dists.shape[0]):
            # Sort dists
            sorted_dists_idx = np.argsort(dists[d])

            closest_idx = sorted_dists_idx[0]
            second_closest_idx = sorted_dists_idx[1]
            # Divide the closest dist with the second closest dist
            if dists[d, second_closest_idx] > 0:
                nndr = dists[d, closest_idx] / dists[d, second_closest_idx]

                if nndr <= self.ratio_threshold:
                    matches.append([d, closest_idx])
                    confidences.append(nndr)

        matches = np.array(matches)
        confidences = np.array(confidences)

        sorted_indices = np.argsort(confidences)
        matches = matches[sorted_indices]
        confidences = confidences[sorted_indices]

        return matches, confidences