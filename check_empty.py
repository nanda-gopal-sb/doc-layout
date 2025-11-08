import os
import json

# Specify the directory containing the JSON files
directory = '.'  # Change this to the path of your directory if needed

empty_annotation_files = []

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'annotations' in data and data['annotations'] == []:
                    empty_annotation_files.append(filename)
        except json.JSONDecodeError:
            print(f"Error decoding {filename}. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping.")

# Output the list of JSON files with empty annotations
print("JSON files with empty annotations:")
for file in empty_annotation_files:
    print(file)
