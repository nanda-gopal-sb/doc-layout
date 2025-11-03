from ultralytics import YOLO

# Load a pre-trained YOLO model (e.g., yolov8n.pt)
# Replace 'path/to/your/model.pt' with the actual path to your model file
model = YOLO('./docLayout.pt')

# Access the dictionary of class names
class_names = model.names

# Print the class names and their corresponding IDs
print("Class IDs and names:")
for class_id, class_name in class_names.items():
    print(f"ID: {class_id}, Name: {class_name}")

# You can also get the total number of classes from the dictionary's length
num_classes = len(class_names)
print(f"\nTotal number of classes: {num_classes}")