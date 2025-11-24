# Technical Report 

Our AI system is built on a multi-stage pipeline architecture designed to intelligently parse and structure content from a wide variety of document formats. The architecture employs distinct processing paths based on the input file type to ensure maximum fidelity.
1. **Format-Specific Ingestion:**
<br>
a. **Digital-Native Path (DOC, DOCX, etc.)**: For textual formats, the system directly inspects the document's internal structure. It parses file metadata and content tags to extract text and layout information, which is then immediately structured into a JSON format.<br>
b. **Image-Based Path (PDF, JPEG, PNG, Handwritten)**: This path handles all image-based or non-native digital files.
PDFs are first rasterized into a series of high-resolution page images using Ghostscript.
These newly created images, along with any JPEG/PNG or scanned handwritten inputs, proceed to the core vision pipeline.
2. **Core Vision and OCR Pipeline:**
Once a document is in image format, it undergoes the following sequential and parallel operations:<br>
    a. **Preprocessing**: The system first analyzes the image for skew. If detected, a deskewing algorithm is applied to straighten the image, which is critical for accurate layout detection and OCR.<br>
    b. **Layout Analysis**: We utilize a YOLO model trained on the DocLayNet dataset to perform document layout analysis. This model localizes and classifies all content elements (e.g., text blocks, tables, figures), exporting their bounding box coordinates to a preliminary JSON file.<br>
    c. **Parallel Multilingual OCR**: Concurrently with the layout analysis, the localized text regions are processed by our OCR engine. To handle mixed scripts (e.g., Hindi-English, English-Arabic), this process involves:
    Language Identification: Using Tesseract, the system first detects the language(s) present in each text block.<br>
    d. **Script Mode Setting**: Based on the identified language, the engine sets the correct reading direction (Left-to-Right or Right-to-Left) to ensure text is extracted in the proper logical order.
4. **Data Fusion and Final Output**: In the final stage, the extracted text and its associated language flags (LTR/RTL) from the OCR engine are programmatically matched and fused with the bounding box data from the YOLO model's JSON. This produces a single, comprehensive JSON file that accurately represents the document's semantic structure, layout, and multilingual content.



# Steps to Run the Model
1. Download the tar file of the docker container.
2. Make sure docker deamon is running and up.
3. `docker load -i polyglot_ai.tar`
4. `docker run --rm -it --gpu all \
  -v "/path/to/images:/data/input" \
  -v "$(pwd)/resutlts:/data/output" \
  polyglot_ai`
