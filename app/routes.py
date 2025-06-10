from flask import Blueprint, render_template, request, current_app, url_for, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import os

from my_utils.classification_utils import classify_car

main = Blueprint('main', __name__)

# Notre route d'accueil pour notre index.html
@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Notre route principale avec notre prédiction et notre fonctionalité de l'historique des prédiction
@main.route('/main', methods=['GET', 'POST'])
def index2():
    prediction = None
    image_path = None
    links = []

    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            saved_path = os.path.join(upload_folder, filename)
            file.save(saved_path)

            # Classification
            image = Image.open(saved_path).convert('RGB')
            prediction = classify_car(image)

            # Ici on gere l'url de l’image pour l'affichage dans <img>
            image_path = f"/static/uploads/{filename}"

            # Nos liens dynamiques après la prédiction
            links = [
                f"https://www.lacentrale.fr/recherche?modele={prediction}",
                f"https://www.leboncoin.fr/recherche?text={prediction}",
                f"https://www.auto-data.net/fr/results?search={prediction}"
            ]

    # L'historique de nos images, ici on les trie du plus récent au plus ancien
    history_images = sorted(
        [f"/static/uploads/{f}" for f in os.listdir(upload_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: os.path.getmtime(os.path.join(upload_folder, os.path.basename(x))),
        reverse=True
    )

    return render_template('main.html',
                           prediction=prediction,
                           image_path=image_path,
                           links=links,
                           history_images=history_images)


# Notre route on effectue la prediction à partir d'une image de l'historique
@main.route('/predict_from_history')
def predict_from_history():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('main.index2'))

    upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
    saved_path = os.path.join(upload_folder, filename)

    if not os.path.exists(saved_path):
        return redirect(url_for('main.index2'))

    # Classification
    image = Image.open(saved_path).convert('RGB')
    prediction = classify_car(image)
    image_path = f"/static/uploads/{filename}"

    links = [
        f"https://www.lacentrale.fr/recherche?modele={prediction}",
        f"https://www.leboncoin.fr/recherche?text={prediction}",
        f"https://www.auto-data.net/fr/results?search={prediction}"
    ]

    # Historique des images comme on le fait dans /main
    history_images = sorted(
        [f"/static/uploads/{f}" for f in os.listdir(upload_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: os.path.getmtime(os.path.join(upload_folder, os.path.basename(x))),
        reverse=True
    )

    return render_template('main.html',
                           prediction=prediction,
                           image_path=image_path,
                           links=links,
                           history_images=history_images)
