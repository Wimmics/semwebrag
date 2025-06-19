from rdflib import Graph, Literal, Namespace, RDF, RDFS
import goslate
import time

def translate_ontology_labels_goslate(input_file, output_file, delay=1):
    """
    Traduit automatiquement les labels anglais en français avec goslate
    
    Args:
        input_file: Chemin vers le fichier d'ontologie d'entrée
        output_file: Chemin vers le fichier de sortie
        delay: Délai entre les requêtes de traduction (en secondes)
    """
    
    # Initialiser le traducteur goslate
    gs = goslate.Goslate()
    
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
            
            # Nettoyer le texte avant traduction
            clean_text = english_text.strip()
            
            # Faire la traduction avec goslate (anglais vers français)
            french_text = gs.translate(clean_text, 'fr')
            
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

def translate_ontology_labels_goslate_batch(input_file, output_file, batch_size=10, delay=2):
    """
    Version optimisée avec traduction par lots pour améliorer les performances
    """
    
    # Initialiser le traducteur goslate
    gs = goslate.Goslate()
    
    # Charger l'ontologie
    print("Chargement de l'ontologie...")
    g = Graph()
    try:
        g.parse(input_file)
        print(f"Ontologie chargée avec {len(g)} triplets")
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return
    
    # Trouver tous les labels anglais qui n'ont pas encore de traduction française
    labels_to_translate = []
    for subject, predicate, obj in g.triples((None, RDFS.label, None)):
        if hasattr(obj, 'language') and obj.language == "en":
            # Vérifier si une traduction française existe déjà
            french_exists = False
            for _, _, existing_obj in g.triples((subject, RDFS.label, None)):
                if hasattr(existing_obj, 'language') and existing_obj.language == "fr":
                    french_exists = True
                    break
            
            if not french_exists:
                labels_to_translate.append((subject, str(obj)))
    
    print(f"Trouvé {len(labels_to_translate)} labels anglais à traduire")
    
    # Traduire par lots
    translated_count = 0
    for i in range(0, len(labels_to_translate), batch_size):
        batch = labels_to_translate[i:i+batch_size]
        print(f"Traduction du lot {i//batch_size + 1}/{(len(labels_to_translate)-1)//batch_size + 1}")
        
        try:
            # Préparer les textes pour la traduction par lot
            texts_to_translate = [text for _, text in batch]
            
            # Traduire le lot
            translated_texts = gs.translate(texts_to_translate, 'fr')
            
            # Si un seul texte, goslate retourne une string, sinon une liste
            if isinstance(translated_texts, str):
                translated_texts = [translated_texts]
            
            # Ajouter les traductions à l'ontologie
            for (subject, original_text), french_text in zip(batch, translated_texts):
                french_label = Literal(french_text, lang="fr")
                g.add((subject, RDFS.label, french_label))
                translated_count += 1
                print(f"  '{original_text}' → '{french_text}'")
            
            # Délai entre les lots
            if i + batch_size < len(labels_to_translate):
                time.sleep(delay)
                
        except Exception as e:
            print(f"Erreur lors de la traduction du lot: {e}")
            # En cas d'erreur, essayer individuellement
            for subject, text in batch:
                try:
                    french_text = gs.translate(text, 'fr')
                    french_label = Literal(french_text, lang="fr")
                    g.add((subject, RDFS.label, french_label))
                    translated_count += 1
                    print(f"  '{text}' → '{french_text}' (individuel)")
                    time.sleep(0.5)
                except:
                    print(f"  Échec pour: '{text}'")
    
    print(f"\nTraduction terminée. {translated_count} labels traduits.")
    
    # Sauvegarder l'ontologie modifiée
    try:
        print(f"Sauvegarde dans {output_file}...")
        
        if output_file.endswith('.ttl'):
            format_output = 'turtle'
        elif output_file.endswith('.rdf') or output_file.endswith('.xml'):
            format_output = 'xml'
        else:
            format_output = 'xml'
            
        g.serialize(destination=output_file, format=format_output)
        print("Sauvegarde terminée avec succès!")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")

def check_translation_coverage(ontology_file):
    """
    Vérifie combien de labels ont déjà des traductions françaises
    """
    g = Graph()
    g.parse(ontology_file)
    
    english_count = 0
    french_count = 0
    entities_with_both = 0
    
    # Grouper par sujet
    entities = {}
    for subject, predicate, obj in g.triples((None, RDFS.label, None)):
        if subject not in entities:
            entities[subject] = {'en': None, 'fr': None}
        
        if hasattr(obj, 'language'):
            if obj.language == 'en':
                entities[subject]['en'] = str(obj)
                english_count += 1
            elif obj.language == 'fr':
                entities[subject]['fr'] = str(obj)
                french_count += 1
    
    # Compter les entités avec les deux langues
    for entity_labels in entities.values():
        if entity_labels['en'] and entity_labels['fr']:
            entities_with_both += 1
    
    print(f"Statistiques de l'ontologie:")
    print(f"- Labels anglais: {english_count}")
    print(f"- Labels français: {french_count}")
    print(f"- Entités avec les deux langues: {entities_with_both}")
    print(f"- Entités nécessitant une traduction: {english_count - entities_with_both}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Vérifier d'abord la couverture actuelle
    print("=== Vérification de la couverture actuelle ===")
    check_translation_coverage("votre_ontologie.owl")
    
    print("\n=== Début de la traduction ===")
    
    # Méthode 1: Traduction une par une (plus sûre)
    translate_ontology_labels_goslate(
        input_file="public-contracts-en.ttl",
        output_file="public-contracts.ttl",
        delay=1
    )
    
    # Méthode 2: Traduction par lots (plus rapide)
    # translate_ontology_labels_goslate_batch(
    #     input_file="votre_ontologie.owl",
    #     output_file="ontologie_traduite.owl",
    #     batch_size=5,
    #     delay=2
    # )