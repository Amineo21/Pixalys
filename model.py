from tensorflow.keras.applications import mobilenet_v2
import numpy as np

# Chargement du modèle MobileNetV2 pré-entraîné
model = mobilenet_v2.MobileNetV2(weights="imagenet")

def predict_image(img_array):
    preds = model.predict(img_array)
    decoded = mobilenet_v2.decode_predictions(preds, top=1)[0]
    label, description, probability = decoded[0]
    return {
        "label": description,
        "probability": float(probability)
    }

def create_google_search_link(label, brand=None, model=None):
    query = f"{brand} {model or label}" if brand else label
    return f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=shop"
