from sentence_transformers import SentenceTransformer
import os

# script pour sauvegarder le modèle d'embedding localement


local_model_path = ".embeddings/models/all-MiniLM-L6-v2"


os.makedirs(local_model_path, exist_ok=True)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save(local_model_path)
print(f"Modèle sauvegardé dans {local_model_path}")