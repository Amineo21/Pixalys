import scipy.io
import json

# Charger le fichier .mat
mat = scipy.io.loadmat('CompCars/misc/make_model_name.mat')

# Extraire les marques et modèles
makes = [str(x[0]) for x in mat['make_names'][0]]
models = [str(x[0]) for x in mat['model_names'][0]]

# Créer une liste : "Renault Clio", "Peugeot 208", etc.
class_names = [f"{make} {model}" for make, model in zip(makes, models)]

# Sauvegarder dans un fichier JSON pour l'utiliser facilement plus tard
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

print(f"{len(class_names)} noms de classes sauvegardés dans class_names.json")
