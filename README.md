# Commandes pour lancer le projet:

## Initialiser flask :
pip install flask
-> au cas où si sa fonctionne pas : flask run 2 fois

## Et ensuite dans un terminal bash :
curl -X POST -F "image=@static/uploads/oiseau.jpg" http://127.0.0.1:5000/predict
-->toute les images sont dans le dossier static/uploads