from flask import Flask, request, jsonify
from model import predict_image
from utils import prepare_image
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return "API d'analyse d'image prête à l'emploi 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    try:
        image = Image.open(file.stream)
        img_array = prepare_image(image)
        result = predict_image(img_array)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
