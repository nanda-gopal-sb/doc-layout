import os
import json
import numpy as np
import cv2
from doclayout_yolo import YOLOv10
# Import the new deskewing module
import deskew

# Initialize the YOLO model
model = YOLOv10("source/docLayout.pt")

# Define the input and output directories
input_dir = "FINAL_DATA/documents/"
output_dir = "output_json/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create a temporary directory for deskewed images
temp_dir = "temp_deskewed_images/"
os.makedirs(temp_dir, exist_ok=True)

category_mapping = {
    0: {"id": 2, "name": "Title"},
    1: {"id": 1, "name": "Text"},
    2: {"id": None, "name": "Abandon"},
    3: {"id": 5, "name": "Figure"},
    4: {"id": 1, "name": "Text"},
    5: {"id": 4, "name": "Table"},
    6: {"id": 1, "name": "Text"},
    7: {"id": 1, "name": "Text"},
    8: {"id": 1, "name": "Text"},
    9: {"id": 1, "name": "Text"}
}

# Iterate through all files in the input directory
for image_file in os.listdir(input_dir):
    # Check if the file is a PNG image
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_dir, image_file)

        # === MODIFIED: Robust Deskewing Code with Exception Handling ===
        image_to_predict_path = image_path  # Default to original image

        try:
            # The deskew function returns the corrected image array or None
            deskewed_image = deskew.deskew_image(image_path, "./out", False, 0)

            # If deskewed_image is not None, it means a skew was detected and corrected.
            if deskewed_image is not None:
                deskewed_image_path = os.path.join(temp_dir, image_file)
                # Save the corrected image to the temp directory
                cv2.imwrite(deskewed_image_path, deskewed_image)
                # Update the path for the YOLO model
                image_to_predict_path = deskewed_image_path
                print(f"Skew detected for {image_file}. Using deskewed image.")
            else:
                print(f"No significant skew detected for {image_file}. Using original image.")

        except Exception as e:
            print(f"An error occurred while deskewing {image_file}: {e}. Falling back to original image.")
        # ===============================================================

        #Perform prediction on the selected image path
        det_res = model.predict(
            image_to_predict_path,
            imgsz=1024,
            conf=0.2,
            device="gpu"
        )
        # Extract bounding box coordinates and class information
        annotations = []
        results = det_res[0]

        for box in results.boxes:
            original_category_id = int(box.cls[0])

            if category_mapping[original_category_id]["id"] is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [
                round(x1, 2),
                round(y1, 2),
                round(x2 - x1, 2),
                round(y2 - y1, 2)
            ]

            new_category = category_mapping[original_category_id]
            new_category_id = new_category["id"]
            new_category_name = new_category["name"]

            annotations.append({
                "bbox": bbox,
                "category_id": new_category_id,
                "category_name": new_category_name
            })

        json_data = {
            "file_name": image_file,
            "annotations": annotations
        }

        output_json_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.json")

        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Bounding box coordinates and new categories saved to {output_json_path}\n")

#Clean up the temporary directory after processing all images
if os.path.exists(temp_dir):
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    print("Temporary deskewed images and directory removed.")
