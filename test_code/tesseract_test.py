import pytesseract
from PIL import Image

# Open the image
img = Image.open("input.png")

# Get bounding box data as a DataFrame
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)

# Filter out empty rows and non-word entries
words_data = data[(data['conf'] != -1) & (data['text'].str.strip() != '')]

# Iterate through the data to access bounding boxes
for index, row in words_data.iterrows():
    x, y, w, h = row['left'], row['top'], row['width'], row['height']
    word = row['text']
    print(f"Word: {word}, Bounding Box: (x={x}, y={y}, w={w}, h={h})")

# To get bounding boxes for lines, blocks, etc., you can adjust the output_type
# For example, to get hOCR output directly:
hocr_output = pytesseract.image_to_hocr(img)
print(hocr_output)
