import os
import cv2
from PIL import Image

def fast_resize(input_folder, output_folder, exif=True):
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
            print(f"Resized and saved with EXIF: {output_path}")
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
                new_width = int(image.shape[1] / 3)
                new_height = int(image.shape[0] / 3)

                # Resize the image
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Transfer EXIF data and save the resized image
                transfer_exif(input_path, resized_image)
            else:
                print(f"Failed to read image: {input_path}")
        else:
            print(f"Skipped non-image file: {filename}")

    print("All images processed.")