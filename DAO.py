import faiss
import numpy as np
import pickle

class FaissDAO:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  
        self.labels = []
   
    def insert(self, label, embedding):
        embedding = np.array(embedding, dtype=np.float32).reshape(1, self.dim)
        self.index.add(embedding)
        self.labels.append(label)

   
   
    def save_index(self, index_file):

        faiss.write_index(self.index, index_file)
        # Save labels to a separate file
        labels_file = index_file + ".labels"
        with open(labels_file, 'wb') as f:
            pickle.dump(self.labels, f)
        print(f"Index sauvegardé dans {index_file}")
        print(f"Labels sauvegardés dans {labels_file}")
   
    def load_index(self, index_file):

        self.index = faiss.read_index(index_file)
        # Load labels from the separate file
        labels_file = index_file + ".labels"
        try:
            with open(labels_file, 'rb') as f:
                self.labels = pickle.load(f)
            print(f"Index chargé depuis {index_file}")
            print(f"Labels chargés depuis {labels_file}")
        except FileNotFoundError:
            print(f"Attention: Fichier de labels {labels_file} non trouvé!")
            print("Les recherches risquent de ne pas fonctionner correctement.")
    
    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, self.dim)
   
        k = min(k, self.index.ntotal)
        if k == 0:
            return [], []
        
        distances, indices = self.index.search(query_embedding, k)
        
        # Check if indices are valid
        results = []
        valid_distances = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.labels):
                results.append(self.labels[idx])
                valid_distances.append(distances[0][i])
        
        return results, valid_distances

    def remove(self, label):

        try:
            # Trouver l'index du label à supprimer
            label_index = self.labels.index(label)
        except ValueError:
            print(f"Label '{label}' non trouvé dans l'index")
            return False
        
        # Récupérer tous les embeddings existants
        embeddings_array = np.zeros((self.index.ntotal, self.dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            embeddings_array[i] = self.index.reconstruct(i)
        
        # Supprimer le label et l'embedding correspondant
        self.labels.pop(label_index)
        embeddings_array = np.delete(embeddings_array, label_index, axis=0)
        
        # Reconstruire l'index FAISS sans l'élément supprimé
        self.index = faiss.IndexFlatL2(self.dim)
        
        if len(embeddings_array) > 0:
            self.index.add(embeddings_array)
        
        print(f"Label '{label}' supprimé avec succès")
        return True
