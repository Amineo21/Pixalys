from PIL import Image
import numpy as np
import pytesseract
from tensorflow.keras.applications import mobilenet_v2

# Spécifie le chemin de Tesseract si nécessaire
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def prepare_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_v2.preprocess_input(img_array)
    return img_array

def extract_brand_text(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""
