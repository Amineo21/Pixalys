FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# On copie les fichiers source
COPY main.py .

# On installe les dépendances
RUN pip install fastapi uvicorn redis rq

# On lance le serveur FastAPI avec uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
