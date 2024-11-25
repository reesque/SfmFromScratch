import numpy as np
from numpy.linalg import svd


class CameraPose:
    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2

    def ransac_camera_motion(self, threshold=1.0, max_iterations=1000):
        best_inliers = []
        best_r, best_t = None, None

        if len(self.pts1) < 8:
            return None, None

        for _ in range(max_iterations):
            # Randomly sample 8 correspondences
            indices = np.random.choice(len(self.pts1), 8, replace=False)
            sample_points1 = self.pts1[indices]
            sample_points2 = self.pts2[indices]

            # Compute the fundamental matrix using the 8-point algorithm
            F = self._compute_fundamental_matrix(sample_points1, sample_points2)

            # Decompose the fundamental matrix to obtain R and T
            # In our case, we assume that the camera intrinsic is the same for both image
            E = F

            # SVD on the essential matrix
            # U and Vt are rotation and reflection, S is scaling
            U, S, Vt = svd(E)

            # Rotation matrix is U and Vt
            W = np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
            R1 = np.dot(U, np.dot(W, Vt))
            R2 = np.dot(U, np.dot(W.T, Vt))

            # Translation vector is the last column of U
            T = U[:, 2]

            # Compute the projection error for all four combinations of R and T
            for R_candidate, T_candidate in [(R1, T), (R1, -T), (R2, T), (R2, -T)]:
                if not self._check_valid_pose(R_candidate, T_candidate):
                    continue  # Skip invalid pose

                inliers = []

                for i in range(len(self.pts1)):
                    x1 = np.array([self.pts1[i, 0], self.pts1[i, 1], 1])  # Homogeneous coordinates for pts1
                    x2 = np.array([self.pts2[i, 0], self.pts2[i, 1], 1])  # Homogeneous coordinates for pts2

                    # Compute the reprojection error using the E
                    error = np.abs(np.dot(x2.T, np.dot(E, x1)))
                    if error < threshold:
                        inliers.append(i)

                # If this rotation and translation give more inliers, select it as the best
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_r, best_t = R_candidate, T_candidate

        return best_r, best_t

    def _compute_fundamental_matrix(self, p1, p2) -> np.ndarray:
        # Ensure points are in homogeneous coordinates
        n = p1.shape[0]
        pts1_hom = np.hstack([p1, np.ones((n, 1))])
        pts2_hom = np.hstack([p2, np.ones((n, 1))])

        # Build the matrix A, where each row corresponds to a correspondence
        # The derivation is as followed:
        # Fundamental equ: l_2 = F.P_1 due to epipolar constraint
        #                  l_2 . P_2 = P2.t.(F.P_1)
        #                  0 = P_2.t.(F.P_1) since P_2 lies on epipolar line l2, hence their dot prod is 0
        # Expand: [x_2 y_2 1] . [[f_11 f_12 f_13]  . [x_1  = 0
        #                        [f_21 f_22 f_23]     y_1
        #                        [f_31 f_32 f_33]]     1]
        # => [x_2 y_2 1] . [[f_11x_1 + f_12y_1 + f_13] = 0
        #                   [f_21x_1 + f_22y_1 + f_23]
        #                   [f_31x_1 + f_32y_1 + f_33]]
        # => f_11x_1x_2 + f_12y_1x_2 + f_13x_2 + f_21x_1y_2 + f_22y_1y_2 + f_23y_2 + f_31x_1 + f_32y_1 + f_33 = 0
        # => [x_1x_2 + y_1x_2 + x_2 + x_1y_2 + x_1y_2 + y_1y_2 + y_2 + x_1 + y_1 + 1] [f_11 = 0
        #                                                                              f_12
        #                                                                              f_13
        #                                                                              f_21
        #                                                                              f_23
        #                                                                              f_31
        #                                                                              f_32
        #                                                                              f_33]
        A = np.zeros((n, 9))
        for i in range(n):
            x1, y1 = pts1_hom[i, 0], pts1_hom[i, 1]
            x2, y2 = pts2_hom[i, 0], pts2_hom[i, 1]
            A[i] = [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]

        # Solve for the vector f (flattened fundamental matrix) using SVD
        _, _, VT = np.linalg.svd(A)
        F_vec = VT[-1, :]

        # Reshape into 3x3 matrix
        F = F_vec.reshape(3, 3)

        # Enforce the rank-2 constraint (set the smallest singular value to zero)
        U, D, Vt = np.linalg.svd(F)
        D[2] = 0  # Set the smallest singular value to 0
        F_rank2 = np.dot(U, np.dot(np.diag(D), Vt))

        return F_rank2

    def _check_valid_pose(self, R_candidate, T_candidate):
        for i in range(len(self.pts1)):
            x1 = np.array([self.pts1[i, 0], self.pts1[i, 1], 1])  # Homogeneous coordinates for pts1
            x2 = np.array([self.pts2[i, 0], self.pts2[i, 1], 1])  # Homogeneous coordinates for pts2

            R1 = np.eye(3)
            t1 = np.zeros(3)

            # Triangulate the 3D point from both images
            X = CameraPose.triangulate_point(x1, x2, R1=R1, t1=t1, R2=R_candidate, t2=T_candidate)

            # Check if the depth (Z) of the triangulated point is positive in both views
            if X[2] < 0:
                return False

        return True

    @staticmethod
    def triangulate_point(x1, x2, R1, t1, R2, t2):
        # Create the projection matrices
        P1 = np.hstack([R1, t1.reshape(-1, 1)])
        P2 = np.hstack([R2, t2.reshape(-1, 1)])

        # Create the system of equations for triangulation
        A = np.vstack([
            x1[0] * P1[2, :] - P1[0, :],
            x1[1] * P1[2, :] - P1[1, :],
            x2[0] * P2[2, :] - P2[0, :],
            x2[1] * P2[2, :] - P2[1, :]
        ])

        # Solve for the 3D point X using SVD
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]  # Normalize to get homogeneous coordinates (X, Y, Z, 1)

        return X[:3]  # Return 3D point (X, Y, Z)
