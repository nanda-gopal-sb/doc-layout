import cv2
import json
import os
import glob

def draw_bounding_boxes(image_path, json_path, output_path):
    """
    Loads an image, draws colored bounding boxes on it based on a JSON
    file, and saves the result.
    """
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"    - Warning: Could not read image from '{image_path}'. Skipping.")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"    - Warning: JSON file not found at '{json_path}'. Skipping.")
        return
    except json.JSONDecodeError:
        print(f"    - Warning: Could not decode JSON from '{json_path}'. Skipping.")
        return

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255)
    ]

    annotations = data.get("annotations")
    if not annotations:
        print(f"    - Info: No annotations found in {json_path}. Saving original image.")
        cv2.imwrite(output_path, image)
        return
        
    unique_categories = sorted(list(set(ann.get('category_id', -1) for ann in annotations)))
    category_colors = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(unique_categories)}

    for annotation in annotations:
        bbox = annotation.get('bbox')
        category_id = annotation.get('category_id')

        if bbox is None or category_id is None:
            continue

        color = category_colors.get(category_id, (255, 255, 255)) # Default to white
        x, y, w, h = map(int, bbox)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
        cv2.putText(image, f"Cat: {category_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(output_path, image)


def process_directories(image_dir, json_dir, output_dir):
    """
    Processes all PNG/JSON pairs from separate input directories and saves the
    results to an output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    search_paths = glob.glob(os.path.join(image_dir, '*.png'))
    search_paths.extend(glob.glob(os.path.join(image_dir, '*.PNG')))
    
    if not search_paths:
        print(f"Error: No .png files found in the directory '{image_dir}'.")
        return

    print(f"\nFound {len(search_paths)} PNG images. Starting processing...")

    for image_path in search_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        json_path = os.path.join(json_dir, base_name + '.json')
    
        output_image_path = os.path.join(output_dir, base_name + '_annotated.png')

        print(f"\nProcessing: {base_name}.png")

        if os.path.exists(json_path):
            draw_bounding_boxes(image_path, json_path, output_image_path)
            print(f"  -> Saved to: {output_image_path}")
        else:
            print(f"  - Warning: Skipping. No matching JSON file found for {os.path.basename(image_path)}")

    print("\n✅ Batch processing complete!")

if __name__ == '__main__':
    images_directory = ''
    json_directory = ''

    output_directory = './test_output'

    process_directories(images_directory, json_directory, output_directory)
