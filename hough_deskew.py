import cv2
import numpy as np
from deskew import determine_skew
import os
import shutil

def deskew(src_img_path):
    """
    Attempts to deskew an image.
    - If skew is detected, returns the rotated image.
    - If no skew is detected, returns the original image.
    - If the image can't be read, throws an exception.
    """
    image = cv2.imread(src_img_path)


    if image is None:
        raise cv2.error(f"Could not read image file: {src_img_path}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)

    if abs(angle) > 0.01:
        print(f"   -> Detected angle: {angle:.2f} degrees")
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            M,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated
    else:
        print("   -> No significant skew detected.")
        return image



directory = "" # specify your input image directory here
out_dir = "deskewed_output" # specify your output directory here

os.makedirs(out_dir, exist_ok=True)

try:
    entries = os.listdir(directory)
except FileNotFoundError:
    print(f"[FATAL ERROR] Input directory not found: {directory}")
    exit()

files = [
    entry
    for entry in entries
    if os.path.isfile(os.path.join(directory, entry))
]

print(f"Found {len(files)} files to process. Output directory: {out_dir}\n")

for file in files:
    src_path = os.path.join(directory, file)
    dest_path = os.path.join(out_dir, file)

    try:
        print(f"Processing: {file}")

        processed_image = deskew(src_path)

        cv2.imwrite(dest_path, processed_image)
        print(f"   -> Successfully saved to: {dest_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {file}: {e}")
        print(f"   -> Copying original (unskewed) file to: {dest_path}")

        try:
            shutil.copy(src_path, dest_path)
        except Exception as copy_e:
            print(f"[FATAL] Could not copy file {src_path}: {copy_e}")

    print("---")

print("\nBatch processing complete.")
