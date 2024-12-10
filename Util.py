import os
import cv2
import numpy as np
from PIL import Image


def fast_resize(input_folder, output_folder, ratio=0.3, exif=True):
    """
    Resize images

    :param input_folder: directory containing the images
    :param output_folder: output directory
    :param ratio: ratio to resize to (both H and W, aspect ratio stays the same)
    :param exif: if EXIF data should be transferred. Set to False if image has no EXIF data
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Function to transfer EXIF data
    def transfer_exif(original_path, resized_image):
        try:
            # Open the original image with Pillow
            original = Image.open(original_path)
            if exif:
                exif_data = original.info.get("exif")

            # Save the resized image with EXIF data using Pillow
            resized_pillow = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            output_path = os.path.join(output_folder, os.path.basename(original_path))
            if exif:
                resized_pillow.save(output_path, format='JPEG', exif=exif_data)
            else:
                resized_pillow.save(output_path, format='JPEG')
            print(f"Resized: {output_path}")
        except Exception as e:
            print(f"Failed to transfer EXIF data for {original_path}: {e}")

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if it's an image file
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image = cv2.imread(input_path)

            # If the image is successfully read
            if image is not None:
                # Calculate the new dimensions
                new_width = int(image.shape[1] * ratio)
                new_height = int(image.shape[0] * ratio)

                # Resize the image
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Transfer EXIF data and save the resized image
                transfer_exif(input_path, resized_image)
            else:
                print(f"Failed to read image: {input_path}")
        else:
            print(f"Skipped non-image file: {filename}")

    print("All images processed.")

def print_reprojection_error(p3d, p1, p2, P1, P2):
    # Compute reprojection error for each triangulated point
    reprojection_errors = []
    for i, (pts1, pts2) in enumerate(zip(p1, p2)):
        X = np.hstack([p3d[i], 1])  # Convert to homogeneous coordinates
        x1_reprojected = P1 @ X
        x2_reprojected = P2 @ X
        x1_reprojected /= x1_reprojected[2]  # Normalize
        x2_reprojected /= x2_reprojected[2]  # Normalize

        # Calculate reprojection error
        error1 = np.linalg.norm(pts1 - x1_reprojected[:2])
        error2 = np.linalg.norm(pts2 - x2_reprojected[:2])
        reprojection_errors.append((error1, error2))

    # Optionally, print or log the mean reprojection error
    mean_error = np.mean([np.mean(errors) for errors in reprojection_errors])
    print(f"Mean Reprojection Error: {mean_error:.4f}")