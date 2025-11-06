from ultralytics import YOLO
model = YOLO('./docLayout.pt')

class_names = model.names

print("Class IDs and names:")
for class_id, class_name in class_names.items():
    print(f"ID: {class_id}, Name: {class_name}")

num_classes = len(class_names)
print(f"\nTotal number of classes: {num_classes}")