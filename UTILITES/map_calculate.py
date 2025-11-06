import json
import numpy as np
import os

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x, y, width, height] format.
    """
    # Convert from [x, y, width, height] to [x1, y1, x2, y2]
    xA1, yA1, wA, hA = boxA
    xB1, yB1, wB, hB = boxB
    
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB2, yB2 = xB1 + wB, yB1 + hB
    
    # Determine the coordinates of the intersection rectangle
    xA = max(xA1, xB1)
    yA = max(yA1, yB1)
    xB = min(xA2, xB2)
    yB = min(yA2, yB2)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = wA * hA
    boxB_area = wB * hB

    # Compute the intersection over union
    iou = interArea / float(boxA_area + boxB_area - interArea)
    return iou

def calculate_ap(recalls, precisions):
    """
    Calculates the Average Precision (AP) using the 11-point interpolation method.
    """
    # Create an array of 11 recall points from 0.0 to 1.0
    recalls_interp = np.linspace(0, 1.0, 11)
    precisions_interp = np.zeros(11)
    
    for i, t in enumerate(recalls_interp):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        precisions_interp[i] = p
        
    return np.mean(precisions_interp)

def run_map_calculation(predictions_path, ground_truth_path, iou_threshold=0.5):
    """
    Main function to load data and calculate mAP.
    """
    try:
        with open(predictions_path, 'r') as f:
            pred_data = json.load(f)
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        return

    # Extract annotations and organize by class
    pred_annotations = {}
    gt_annotations = {}
    unique_classes = set()

    for pred in pred_data.get('annotations', []):
        class_id = pred['category_id']
        unique_classes.add(class_id)
        if class_id not in pred_annotations:
            pred_annotations[class_id] = []
        # Your generated JSON does not have a confidence score.
        # We assign a default of 1.0 for all predictions.
        pred_annotations[class_id].append({
            'bbox': pred['bbox'],
            'confidence': 1.0
        })

    for gt in gt_data.get('annotations', []):
        class_id = gt['category_id']
        unique_classes.add(class_id)
        if class_id not in gt_annotations:
            gt_annotations[class_id] = []
        gt_annotations[class_id].append({
            'bbox': gt['bbox'],
            'matched': False
        })
    
    # Calculate AP for each class
    average_precisions = []
    
    for class_id in sorted(list(unique_classes)):
        preds = sorted(pred_annotations.get(class_id, []), key=lambda x: x['confidence'], reverse=True)
        gts = gt_annotations.get(class_id, [])
        num_gt = len(gts)
        
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))
        
        # Match predictions to ground truths
        for i, pred in enumerate(preds):
            iou_max = -1
            best_gt_idx = -1
            
            for j, gt in enumerate(gts):
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    best_gt_idx = j
            
            if iou_max >= iou_threshold and not gts[best_gt_idx]['matched']:
                tps[i] = 1
                gts[best_gt_idx]['matched'] = True
            else:
                fps[i] = 1
        
        # Compute cumulative sums
        cum_fps = np.cumsum(fps)
        cum_tps = np.cumsum(tps)
        
        # Compute precision and recall
        if num_gt == 0:
            recalls = np.zeros(len(preds))
        else:
            recalls = cum_tps / num_gt
        
        precisions = cum_tps / (cum_fps + cum_tps)
        
        ap = calculate_ap(recalls, precisions)
        average_precisions.append(ap)
        
        print(f"AP for class {class_id} (num_gt={num_gt}): {ap:.4f}")

    # Calculate mAP
    if average_precisions:
        mean_ap = np.mean(average_precisions)
        print(f"\nMean Average Precision (mAP) @ IoU={iou_threshold}: {mean_ap:.4f}")
        return mean_ap
    else:
        print("\nNo annotations found in either file to calculate mAP.")
        return 0

# Example usage:
if __name__ == "__main__":
    predictions_file = 'result.json'  # Replace with your predicted JSON file path
    ground_truth_file = 'doc_02491.json' # Replace with your ground truth JSON file path
    
    run_map_calculation(predictions_file, ground_truth_file)