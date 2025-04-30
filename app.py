from flask import Flask, request, jsonify, render_template
from model import predict_image, create_google_search_link
from utils import prepare_image, extract_brand_text
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Nom de fichier vide"}), 400

        try:
            image = Image.open(file.stream)

            # Étape 1 : OCR pour deviner la marque
            brand = extract_brand_text(image)

            # Étape 2 : Prédiction de l'objet
            img_array = prepare_image(image)
            prediction = predict_image(img_array)

            # Étape 3 : Génération du lien Google d'achat
            search_url = create_google_search_link(
                label=prediction["label"],
                brand=brand if brand else None
            )

            return render_template("index.html", label=prediction["label"], brand=brand, probability=prediction["probability"], buy_link=search_url)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")
