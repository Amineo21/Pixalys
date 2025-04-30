# Commandes pour lancer le projet:

## Installer les dépendances:
->Créer un fichier requirements.txt et y mettre ceci :
flask
tensorflow
pillow
numpy
pytesseract

->Puis entrer dans le terminal shell cet commande :
 pip install -r requirements.txt   

-> Au cas où flask ne fonctionne pas, on peut l'installer avec : 
pip install flask

## Lancer le projet :
->Dans un terminal shell :
flask run (2fois si __pycache n'est pas présent lors de la première installation)

 ->Et ensuite dans un terminal bash pour tester l'IA :
curl -X POST -F "image=@static/uploads/oiseau.jpg" http://127.0.0.1:5000/predict
-->toute les images sont dans le dossier static/uploads