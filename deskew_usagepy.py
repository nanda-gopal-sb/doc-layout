import os
from deskew import deskew_image

directory = "/home/nandagopal/Projects/aiGrantz/MOCK_DATA/documents/"
entries = os.listdir(directory)

files = [
    entry
    for entry in entries
    if os.path.isfile(os.path.join(directory, entry))
]

for file in files:
    print(f"Processing: {file}")
    deskew_image(directory + file, "./out", True, 0)
    print("\n\n")

# fails in these case:
# doc_03818.png
# doc_03163.png
deskew_image(directory + "doc_03163.png")