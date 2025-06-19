# généré chatgpt

from rdflib import Graph, Literal, Namespace, RDF, RDFS
from googletrans import Translator
import time
import re

def translate_ontology_labels(input_file, output_file, delay=1):
    """
    Traduit automatiquement les labels anglais en français dans une ontologie OWL/RDF
    
    Args:
        input_file: Chemin vers le fichier d'ontologie d'entrée
        output_file: Chemin vers le fichier de sortie
        delay: Délai entre les requêtes de traduction (en secondes)
    """
    
    # Initialiser le traducteur
    translator = Translator()
    
    # Charger l'ontologie
    print("Chargement de l'ontologie...")
    g = Graph()
    try:
        g.parse(input_file)
        print(f"Ontologie chargée avec {len(g)} triplets")
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return
    
    # Trouver tous les labels anglais
    english_labels = []
    for subject, predicate, obj in g.triples((None, RDFS.label, None)):
        if hasattr(obj, 'language') and obj.language == "en":
            english_labels.append((subject, str(obj)))
    
    print(f"Trouvé {len(english_labels)} labels anglais à traduire")
    
    # Traduire et ajouter les labels français
    translated_count = 0
    for subject, english_text in english_labels:
        try:
            # Vérifier si une traduction française existe déjà
            french_exists = False
            for _, _, existing_obj in g.triples((subject, RDFS.label, None)):
                if hasattr(existing_obj, 'language') and existing_obj.language == "fr":
                    french_exists = True
                    break
            
            if french_exists:
                print(f"Traduction française déjà existante pour: {english_text}")
                continue
            
            # Traduire le texte
            print(f"Traduction de: '{english_text}'")
            
            # Nettoyer le texte avant traduction (enlever caractères spéciaux si nécessaire)
            clean_text = english_text.strip()
            
            # Faire la traduction
            translation = translator.translate(clean_text, src='en', dest='fr')
            french_text = translation.text
            
            # Ajouter le label français à l'ontologie
            french_label = Literal(french_text, lang="fr")
            g.add((subject, RDFS.label, french_label))
            
            translated_count += 1
            print(f"  → '{french_text}'")
            
            # Délai pour éviter de surcharger l'API de traduction
            time.sleep(delay)
            
        except Exception as e:
            print(f"Erreur lors de la traduction de '{english_text}': {e}")
            continue
    
    print(f"\nTraduction terminée. {translated_count} labels traduits.")
    
    # Sauvegarder l'ontologie modifiée
    try:
        print(f"Sauvegarde dans {output_file}...")
        
        # Déterminer le format de sortie basé sur l'extension
        if output_file.endswith('.ttl'):
            format_output = 'turtle'
        elif output_file.endswith('.rdf') or output_file.endswith('.xml'):
            format_output = 'xml'
        elif output_file.endswith('.n3'):
            format_output = 'n3'
        else:
            format_output = 'xml'  # par défaut
            
        g.serialize(destination=output_file, format=format_output)
        print("Sauvegarde terminée avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")

def translate_ontology_labels_with_deepl(input_file, output_file, deepl_api_key, delay=1):
    """
    Version utilisant DeepL API (plus précise mais nécessite une clé API)
    """
    import deepl
    
    # Initialiser DeepL
    translator = deepl.Translator(deepl_api_key)
    
    # Charger l'ontologie
    print("Chargement de l'ontologie...")
    g = Graph()
    try:
        g.parse(input_file)
        print(f"Ontologie chargée avec {len(g)} triplets")
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return
    
    # Trouver tous les labels anglais
    english_labels = []
    for subject, predicate, obj in g.triples((None, RDFS.label, None)):
        if hasattr(obj, 'language') and obj.language == "en":
            english_labels.append((subject, str(obj)))
    
    print(f"Trouvé {len(english_labels)} labels anglais à traduire")
    
    # Traduire et ajouter les labels français
    translated_count = 0
    for subject, english_text in english_labels:
        try:
            # Vérifier si une traduction française existe déjà
            french_exists = False
            for _, _, existing_obj in g.triples((subject, RDFS.label, None)):
                if hasattr(existing_obj, 'language') and existing_obj.language == "fr":
                    french_exists = True
                    break
            
            if french_exists:
                continue
            
            # Traduire avec DeepL
            print(f"Traduction de: '{english_text}'")
            result = translator.translate_text(english_text, target_lang="FR")
            french_text = result.text
            
            # Ajouter le label français
            french_label = Literal(french_text, lang="fr")
            g.add((subject, RDFS.label, french_label))
            
            translated_count += 1
            print(f"  → '{french_text}'")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"Erreur lors de la traduction de '{english_text}': {e}")
            continue
    
    print(f"\nTraduction terminée. {translated_count} labels traduits.")
    
    # Sauvegarder
    try:
        g.serialize(destination=output_file, format='xml')
        print("Sauvegarde terminée!")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Utilisation avec Google Translate (gratuit)
    translate_ontology_labels(
        input_file="callForTender/public-contracts-en.ttl",
        output_file="callForTender/public-contracts.ttl",
        delay=0.5  # 1 seconde entre chaque traduction
    )
    
    # Ou avec DeepL (plus précis, nécessite une clé API)
    # translate_ontology_labels_with_deepl(
    #     input_file="votre_ontologie.owl", 
    #     output_file="ontologie_avec_francais.owl",
    #     deepl_api_key="votre_cle_deepl"
    # )