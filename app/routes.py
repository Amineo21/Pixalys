from flask import Blueprint, render_template, request, current_app
from werkzeug.utils import secure_filename
from PIL import Image
import os

from my_utils.classification_utils import classify_car

main = Blueprint('main', __name__)

# Route d'accueil pour index.html (page de couverture)
@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route principale avec la prédiction
@main.route('/main', methods=['GET', 'POST'])
def index2():
    prediction = None
    image_path = None
    links = []

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            saved_path = os.path.join(upload_folder, filename)
            file.save(saved_path)

            # Classification
            image = Image.open(saved_path).convert('RGB')
            prediction = classify_car(image)

            # URL de l’image pour affichage dans <img>
            image_path = f"/static/uploads/{filename}"

            # Liens dynamiques
            links = [
                f"https://www.lacentrale.fr/recherche?modele={prediction}",
                f"https://www.leboncoin.fr/recherche?text={prediction}",
                f"https://www.auto-data.net/fr/results?search={prediction}"
            ]

    return render_template('main.html', prediction=prediction, image_path=image_path, links=links)
