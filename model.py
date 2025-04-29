import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
import numpy as np

# Charger un modèle MobileNetV2 pré-entraîné
model = mobilenet_v2.MobileNetV2(weights="imagenet")

def predict_image(img_array):
    """Fait une prédiction sur une image prétraitée"""
    preds = model.predict(img_array)
    decoded = mobilenet_v2.decode_predictions(preds, top=1)[0]
    label, description, probability = decoded[0]
    return {
        "label": description,
        "probability": float(probability)
    }