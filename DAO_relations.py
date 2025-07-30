import faiss
import numpy as np
import pickle

class FaissDAO_relations:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  
        self.labels = []
        self.sources = []
        self.destinations = []
   
    def insert(self, label, source, destination, embedding):

        embedding = np.array(embedding, dtype=np.float32).reshape(1, self.dim)
        self.index.add(embedding)
        self.labels.append(label)
        self.sources.append(source)
        self.destinations.append(destination)
   
    def save_index(self, index_file):
        faiss.write_index(self.index, index_file)
        
        # Sauvegarde des métadonnées dans un fichier séparé
        metadata_file = index_file + ".metadata"
        metadata = {
            'labels': self.labels,
            'sources': self.sources,
            'destinations': self.destinations
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Index sauvegardé dans {index_file}")
        print(f"Métadonnées sauvegardées dans {metadata_file}")
   
    def load_index(self, index_file):
        self.index = faiss.read_index(index_file)
        
        metadata_file = index_file + ".metadata"
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                
            self.labels = metadata['labels']
            self.sources = metadata['sources']
            self.destinations = metadata['destinations']
            
            print(f"Index chargé depuis {index_file}")
            print(f"Métadonnées chargées depuis {metadata_file}")
            
        except FileNotFoundError:
            print(f"Attention: Fichier de métadonnées {metadata_file} non trouvé!")
            print("Les recherches risquent de ne pas fonctionner correctement.")
            self.labels = []
            self.sources = []
            self.destinations = []
   
    # def search(self, query_embedding, k=5):

    #     query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, self.dim)
   
    #     k = min(k, self.index.ntotal)
    #     if k == 0:
    #         return [], []
       
    #     distances, indices = self.index.search(query_embedding, k)
       
    #     relations = []
    #     valid_distances = []
        
    #     for i, idx in enumerate(indices[0]):
    #         if 0 <= idx < len(self.labels):
    #             relation = {
    #                 'source': self.sources[idx],
    #                 'destination': self.destinations[idx],
    #                 'label': self.labels[idx]
    #             }
    #             relations.append(relation)
    #             valid_distances.append(distances[0][i])
       
    #     return relations, valid_distances
    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, self.dim)
        
        if self.index.ntotal == 0:
            return [], []
        
        # Commencer par chercher k résultats
        search_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)
        
        relations = []
        valid_distances = []
        seen_labels = set()
        
        # Traiter les premiers résultats
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.labels):
                label = self.labels[idx]
                relation = {
                    'source': self.sources[idx],
                    'destination': self.destinations[idx],
                    'label': label
                }
                relations.append(relation)
                valid_distances.append(distances[0][i])
                seen_labels.add(label)
        
        # Si on n'a pas assez de labels différents, chercher plus loin
        while len(seen_labels) < k and search_k < self.index.ntotal:
            # Augmenter la taille de recherche
            search_k = min(search_k * 2, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, search_k)
            
            # Recommencer le traitement avec plus de résultats
            relations = []
            valid_distances = []
            seen_labels = set()
            
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.labels):
                    label = self.labels[idx]
                    relation = {
                        'source': self.sources[idx],
                        'destination': self.destinations[idx],
                        'label': label
                    }
                    relations.append(relation)
                    valid_distances.append(distances[0][i])
                    seen_labels.add(label)
                    
                    # Arrêter si on a k labels différents
                    if len(seen_labels) >= k:
                        break
        
        return relations, valid_distances
    
    # def search_specific(self, query_embedding, entity_label, k=5):
    #     """
    #     Recherche les meilleures relations qui contiennent le label d'entité spécifié
    #     soit en source soit en destination
        
    #     Args:
    #         query_embedding: Embedding de la requête
    #         entity_label (str): Label d'entité à rechercher en source ou destination
    #         k (int): Nombre de résultats à retourner (défaut: 5)
            
    #     Returns:
    #         tuple: (relations, distances) - listes des relations et distances correspondantes
    #     """
    #     query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, self.dim)
        
    #     if self.index.ntotal == 0:
    #         return [], []
        
    #     # Trouver tous les indices qui correspondent au critère
    #     matching_indices = []
    #     for i, (source, destination) in enumerate(zip(self.sources, self.destinations)):
    #         if source.lower() in entity_label.lower() or destination.lower() in entity_label.lower():
    #             matching_indices.append(i)
        
    #     if not matching_indices:
    #         # Aucune relation trouvée pour cette entité
    #         return [], []
        
    #     # Chercher parmi tous les embeddings pour avoir les distances
    #     search_k = min(self.index.ntotal, max(k * 10, 100))  # Chercher plus large au début
    #     distances, indices = self.index.search(query_embedding, search_k)
        
    #     # Filtrer pour ne garder que les indices qui matchent notre critère
    #     filtered_results = []
    #     for i, idx in enumerate(indices[0]):
    #         if idx in matching_indices:
    #             filtered_results.append((distances[0][i], idx))
        
    #     # Trier par distance et prendre les k meilleurs
    #     filtered_results.sort(key=lambda x: x[0])
    #     filtered_results = filtered_results[:k]
        
    #     # Construire les résultats finaux
    #     relations = []
    #     valid_distances = []
        
    #     for distance, idx in filtered_results:
    #         relation = {
    #             'source': self.sources[idx],
    #             'destination': self.destinations[idx],
    #             'label': self.labels[idx]
    #         }
    #         relations.append(relation)
    #         valid_distances.append(distance)
        
    #     return relations, valid_distances

   
    def search_specific(self, query_embedding, entity_label, k=5):
        """
        Recherche les meilleures relations qui contiennent le label d'entité spécifié
        soit en source soit en destination
        
        Args:
            query_embedding: Embedding de la requête
            entity_label (str): Label d'entité à rechercher en source ou destination
            k (int): Nombre de résultats à retourner (défaut: 5)
            
        Returns:
            tuple: (relations, distances) - listes des relations et distances correspondantes
        """
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, self.dim)
        
        if self.index.ntotal == 0:
            return [], []
        
        # Trouver tous les indices qui correspondent au critère
        matching_indices = set()
        for i, (source, destination) in enumerate(zip(self.sources, self.destinations)):
            if entity_label.lower() == source.lower() or entity_label.lower() == destination.lower():
                matching_indices.add(i)
        
        if not matching_indices:
            # Aucune relation trouvée pour cette entité
            print(f"Aucune relation trouvée pour l'entité '{entity_label}'")
            return [], []
        
        print(f"Trouvé {len(matching_indices)} relations pour l'entité '{entity_label}'")
        
        # Chercher parmi tous les embeddings pour avoir les distances
        search_k = self.index.ntotal  # Chercher dans tout l'index
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Filtrer pour ne garder que les indices qui matchent notre critère
        filtered_results = []
        for i, idx in enumerate(indices[0]):
            if idx in matching_indices:
                filtered_results.append((distances[0][i], idx))
        
        # Trier par distance et prendre les k meilleurs
        filtered_results.sort(key=lambda x: x[0])
        filtered_results = filtered_results[:k]
        
        # Construire les résultats finaux
        relations = []
        valid_distances = []
        
        for distance, idx in filtered_results:
            relation = {
                'source': self.sources[idx],
                'destination': self.destinations[idx],
                'label': self.labels[idx]
            }
            relations.append(relation)
            valid_distances.append(distance)
        
        return relations, valid_distances


    def remove(self, label):

        try:
            # Trouve l'index du label à supprimer
            label_index = self.labels.index(label)
        except ValueError:
            print(f"Label '{label}' non trouvé dans l'index")
            return False
       
        # Récupère tous les embeddings
        embeddings_array = np.zeros((self.index.ntotal, self.dim), dtype=np.float32)
        for i in range(self.index.ntotal):
            embeddings_array[i] = self.index.reconstruct(i)
       
        # Supprime le label et ses métadonnées correspondantes
        self.labels.pop(label_index)
        self.sources.pop(label_index)
        self.destinations.pop(label_index)
        embeddings_array = np.delete(embeddings_array, label_index, axis=0)
       
        # Recrée l'index avec les embeddings restants
        self.index = faiss.IndexFlatL2(self.dim)
       
        if len(embeddings_array) > 0:
            self.index.add(embeddings_array)
       
        print(f"Relation '{label}' supprimée avec succès")
        return True
    
    def get_all_relations(self):
        """
        Retourne toutes les relations stockées
        
        Returns:
            list: Liste de dictionnaires contenant toutes les relations
        """
        relations = []
        for i in range(len(self.labels)):
            relation = {
                'source': self.sources[i],
                'destination': self.destinations[i],
                'label': self.labels[i]
            }
            relations.append(relation)
        return relations
    
    def find_by_source(self, source):
        """
        Trouve toutes les relations ayant une source donnée
        
        Args:
            source (str): Source à rechercher
            
        Returns:
            list: Liste des relations correspondantes
        """
        relations = []
        for i, src in enumerate(self.sources):
            if src == source:
                relation = {
                    'source': self.sources[i],
                    'destination': self.destinations[i],
                    'label': self.labels[i]
                }
                relations.append(relation)
        return relations
    
    def find_by_destination(self, destination):
        """
        Trouve toutes les relations ayant une destination donnée
        
        Args:
            destination (str): Destination à rechercher
            
        Returns:
            list: Liste des relations correspondantes
        """
        relations = []
        for i, dest in enumerate(self.destinations):
            if dest == destination:
                relation = {
                    'source': self.sources[i],
                    'destination': self.destinations[i],
                    'label': self.labels[i]
                }
                relations.append(relation)
        return relations
    
    def __len__(self):
        """Retourne le nombre de relations stockées"""
        return len(self.labels)
    
    def __str__(self):
        """Représentation string de l'objet"""
        return f"FaissDAO_relations({len(self)} relations, dim={self.dim})"