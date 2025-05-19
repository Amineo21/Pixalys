# 🚗 Vehicle Recognition with ResNet50 and CompCars

This project trains a fine-tuned ResNet50 model on the [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) dataset to recognize car **make and model** from images.

## 📁 Project Structure


CompCars/
├── Image/
│   ├── ...
├── ...
models/
├── marque_modèle_2
├── marque_modèle_2
class_name.json
└── ...


## ⚙️ Setup

1. Clone le projet et Télécharge le dataset CompCars : https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

Petite Precision : Le dataset CompCars est divisé en deux parties :​
* Web-nature : Images provenant du web, avec différentes vues des véhicules.
* Surveillance-nature : Images capturées par des caméras de surveillance, principalement en vue frontale.
Pour ce projet vous avez besoin de télécharger uniquement la partie Web-nature ( donc tous les data.z* et le data.zip uniquement )


2. Installe les dépendances : pip install torch torchvision pillow scipy numpy matplotlib


## 🎯 **Pour entrainer le model**

Petite Precision : Le model a déjà été entrainé une première fois en effectuant Python Train.py vous améliorer le modèle en entrainant les dernières couches du modèle pré entrainé.

python clean.py
python train.py

## 🥇 **Pour Tester le model** 

python test_model.py 
