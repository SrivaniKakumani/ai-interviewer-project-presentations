# ocr_utils.py
import io
import cv2
import numpy as np
import easyocr
from PIL import Image

# Create reader once (English only – add languages if needed, e.g. ['en', 'hi'])
reader = easyocr.Reader(['en'], gpu=False)

def image_bytes_to_text(image_bytes: bytes) -> str:
    """
    Take raw image bytes (from Streamlit uploader) and return extracted text.
    """
    # Load with PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Convert to OpenCV format
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run EasyOCR
    results = reader.readtext(img_bgr, detail=0)  # detail=0 -> only text strings
    # Join all pieces into one text block
    return "\n".join(results)
