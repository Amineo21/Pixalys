from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def prepare_image(image, target_size=(224, 224)):
    """Prépare l’image pour MobileNet"""
    image = image.resize(target_size)
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
def load_image(path):
    """Charge l'image"""
    img = Image.open(path)
    return img