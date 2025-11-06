import os
import json
import sys
import time
from doclayout_yolo import YOLOv10
import deskew_clustering

INPUT_DIR = "" # specify your input image directory here
JSON_OUTPUT_DIR = "output_json/"

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

def print_flush(x):
    print(x)
    sys.stdout.flush()

def process(img_filename, img_path, model, json_dir):
    deskewed_image = deskew_clustering.deskew_image(img_path)
    det_res = model.predict(
        deskewed_image,
        imgsz=1024,
        conf=0.2,
        device="cpu",
        verbose=False
    )
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
    save_json_file({
        "file_name": img_filename,
        "annotations": annotations
    }, json_dir)


def save_json_file(data, out_path):
    fn = os.path.splitext(data["file_name"])[0]
    output_json_path = os.path.join(out_path, fn + ".json")
    # print("saving file: "+ fn + ".json")
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Initialize the YOLO model
    model = YOLOv10("models/docLayout.pt")
    print_flush("getting files\n")
    files = os.listdir(INPUT_DIR)
    count = len(files)
    # Create output directory if it doesn't exist
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    start = time.perf_counter()
    for i, img_filename in enumerate(files):
        now = time.perf_counter() - start
        eta = now/(i+1) * (count-i+1)
        print(
f"""Elapsed     : {(now-now%60)/60:2.0f}m:{(now%60):2.3f}s
time per img: {now/(i+1)*1000:.0f} ms
ETA         : {(eta - eta%60)/60:2.0f}m:{eta%60:2.0f}s

processing image: {img_filename}  {i+1}/{count}""")
        img_path = os.path.join(INPUT_DIR, img_filename)
        process(img_filename, img_path, model, JSON_OUTPUT_DIR)
        sys.stdout.write("\033[5A")  # move cursor up 5 lines
        sys.stdout.write("\033[J")   # clear from cursor to end of screen
