import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from doclayout_yolo import YOLOv10
from run import process, print_flush

IMG_DIR = "PS05_SHORTLIST_DATA/images"
JSON_OUTPUT_DIR = "output_json/"

# Create output directory if it doesn't exist
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

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
    9: {"id": 1, "name": "Text"},
}



def proc(img_dir, img_filename, json_dir, i, count):
    model = YOLOv10("models/docLayout.pt")
    print_flush(f"processing image {i+1}/{count}")
    img_path = os.path.join(img_dir, img_filename)
    process(img_filename, img_path, model, json_dir)


if __name__ == "__main__":
    print_flush("getting files\n")
    files = os.listdir(IMG_DIR)
    count = len(files)
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                proc, IMG_DIR, img_filename, JSON_OUTPUT_DIR, i, count
            )
            for i, img_filename in enumerate(files)
        ]