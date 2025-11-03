from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Define the local directory to save models
DOWNLOAD_PATH = Path("./models")
DOWNLOAD_PATH.mkdir(exist_ok=True)

# Choose which model to use
# 0: nano, 1: small, 2: medium
model_files = [
    "yolo11n_doc_layout.pt",
    "yolo11s_doc_layout.pt",
    "yolo11m_doc_layout.pt",
]
selected_model_file = model_files[0] # Using the recommended nano model

# Download the model from the Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Armaggheddon/yolo11-document-layout",
    filename=selected_model_file,
    repo_type="model",
    local_dir=DOWNLOAD_PATH,
)

# Initialize the YOLO model
model = YOLO(model_path)

# Run inference on an image
# Replace 'path/to/your/document.jpg' with your file
results = model('/home/nandagopal/Projects/aiGrantz/MOCK_DATA/documents/doc_00097.png')

# Process and display results
 # Print detection details
results[0].show()   # Display the image with bounding boxes

