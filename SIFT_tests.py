import numpy as np
import unittest
from SIFT import SIFT

class TestSIFT(unittest.TestCase):
    
    def setUp(self):
        self.sift = SIFT()
    
    def test_compute_image_gradients(self, image_bw):
        Ix, Iy = self.sift._compute_image_gradients(image_bw)
        
        # Check that gradients are of correct dimensions
        self.assertEqual(Ix.shape, image_bw.shape)
        self.assertEqual(Iy.shape, image_bw.shape)

    def test_generate_gaussian_kernel(self, ksize, sigma):
        kernel = self.sift._generate_gaussian_kernel(ksize, sigma)
        
        # Check kernel dimensions and normalization
        self.assertEqual(kernel.shape, (ksize, ksize))
        self.assertAlmostEqual(np.sum(kernel), 1, places=5)

    def test_find_harris_interest_points(self, image_bw):
        x, y, confidences = self.sift.find_harris_interest_points(image_bw)
        
        # Ensure arrays are non-empty and match in length
        self.assertTrue(len(x) > 0)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(confidences))
    
    def test_get_SIFT_descriptors(self, image_bw, x, y):
        descriptors = self.sift.get_SIFT_descriptors(image_bw, x, y, feature_width=16)
        
        # Verify descriptor shape and dimensionality
        self.assertEqual(descriptors.shape[1], 128)