import requests
import re
from difflib import SequenceMatcher
import rdflib
import os
import json
import time
from rdflib import Namespace, RDF, OWL, RDFS, URIRef, Literal

def get_uri_wikidata(entite, langue="en"): # doit forcement retourner une URI wikidata

    # recherche directement avec le mot tel quel
    uri = recherche_directe(entite, langue)
    if uri:
        return uri
    
    # sinon essayer des transformations
    transformations = generer_transformations(entite)
    for terme in transformations:
        uri = recherche_directe(terme, langue)
        if uri:
            return uri
    
    
    # en dernier recours recherche des mots-clés les plus longs
    mots = re.findall(r'\b[A-Za-z]{3,}\b', entite)
    mots.sort(key=len, reverse=True)  # longueur décroissante
    
    for mot in mots[:3]:  # Essayer les 3 mots les plus longs
        uri = recherche_directe(mot, langue)
        if uri:
            return uri
    
    # sinon retourner une entité par défaut
    if re.search(r'\d+', entite):  #contient des chiffres
        return "http://www.wikidata.org/wiki/Q12503"  # Q12503 = nombre 
    else:
        return "http://www.wikidata.org/wiki/Q35120"  # Q35120 = entité

def recherche_directe(terme, langue="fr"):

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": terme,
        "language": langue,
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if "search" in data and len(data["search"]) > 0:
            entity_id = data["search"][0]["id"]
            return f"http://www.wikidata.org/wiki/{entity_id}"
        return None
    except Exception as e:
        print(f"Erreur lors de la recherche: {e}")
        return None

def generer_transformations(entite):

    transformations = []

    # enlever les chiffres
    sans_chiffres = re.sub(r'\d+\s*', '', entite).strip()
    if sans_chiffres and sans_chiffres != entite:
        transformations.append(sans_chiffres)
    
    # pas de chiffre pas de ponctuatio
    mots_seulement = re.sub(r'[^a-zA-Z\s]', ' ', entite).strip()
    mots_seulement = re.sub(r'\s+', ' ', mots_seulement)  # Remplacer les espaces multiples
    if mots_seulement and mots_seulement != entite and mots_seulement not in transformations:
        transformations.append(mots_seulement)
    
    # minuscules
    if entite.lower() != entite:
        transformations.append(entite.lower())
    
    # enlever mots communs
    stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'le', 'la', 'les', 'un', 'une', 'des']
    mots = entite.lower().split()
    sans_stop_words = ' '.join(mot for mot in mots if mot not in stop_words)
    if sans_stop_words and sans_stop_words != entite.lower() and sans_stop_words not in transformations:
        transformations.append(sans_stop_words)
    
    return transformations



import requests

def get_description_from_entity(entity_url):
    # Extraire l'ID de l'entité depuis l'URL
    entity_id = entity_url.split('/')[-1]  # Récupère la partie après le dernier '/'
    
    # URL pour l'API de Wikidata
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
       
        entity_data = data.get('entities', {}).get(entity_id, {})
        labels = entity_data.get('labels', {})
        
    
        if 'en' in labels:
            return labels['en'].get('value')
        else:
            return None  
    else:
        return None  



def retrieve_mentioned_chunks(graph_path, entity, chunk_already_Mentionned, neighborChunks):
    #take a graph and an entity and return the chunks that are mentioned in the graph (directly or via neigboors)
    l = []
    graph = rdflib.Graph()
    graph.parse(graph_path, format='turtle')  
    
    mentioned_chunks = set() 

    query = """
    SELECT ?entity ?chunk ?label WHERE {
    {
    
        ?entity rel:mentionedIn ?chunk .
        ?chunk skos:prefLabel ?label .
        FILTER (CONTAINS(LCASE(str(?label)), LCASE(str(?entity_label))))
    }
    UNION
    {
        ?entity ex:isWikidataNeighborOf ?wikidata_uri .
        ?entity skos:prefLabel ?label2 .
        ?mentioned_entity a ?wikidata_uri .
        
        ?mentioned_entity rel:mentionedIn ?chunk .
        ?chunk skos:prefLabel ?label .
        FILTER (CONTAINS(LCASE(str(?label2)), LCASE(str(?entity_label))))
    }
    }
    """


    results = graph.query(query, initBindings={'entity_label': entity})

    base_chunks = []

    for row in results:
        chunk_uri = str(row.chunk)
        label = str(row.label)

        print(f"Entity: {row.entity}, Chunk: {chunk_uri}") 

        if label not in l:
            l.append(label)

        if label not in chunk_already_Mentionned:
            chunk_already_Mentionned.append(label)

        base_chunks.append(chunk_uri)

    # Ajouter les chunks voisins
    if neighborChunks > 0:
        pattern = re.compile(r'(.*chunk_)(\d+)$')
        for uri in base_chunks:
            match = pattern.match(uri)
            if match:
                prefix, number = match.groups()
                number = int(number)
                for i in range(1, neighborChunks + 1):
                    for neighbor_num in [number - i, number + i]:
                        neighbor_uri = f"{prefix}{neighbor_num}"
                        if neighbor_uri not in base_chunks and neighbor_uri not in mentioned_chunks:
                            mentioned_chunks.add(neighbor_uri)

    # Ajouter les voisins à la liste si leur label n'est pas déjà dedans
    for uri in mentioned_chunks:
        # Récupérer le label du chunk voisin depuis le graphe
        query_label = """
        SELECT ?label WHERE {
            <""" + uri + """> skos:prefLabel ?label .
        }
        """
        results_label = graph.query(query_label)
        for row in results_label:
            label = str(row.label)
            # if label not in l:
            l.append(label)
            if label not in chunk_already_Mentionned:
                chunk_already_Mentionned.append(label)

        print

    return l, chunk_already_Mentionned



def add_text_to_file(filepath, text) : # ajouter un texte à un fichier txt sans écraser le contenu
    with open(filepath, 'a') as file:
        file.write(text)



def initialize_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as json_file:
            json.dump([], json_file)  # liste vide
    else:
        with open(file_path, 'w') as json_file:
            json.dump([], json_file) 

def add_to_json_file(file_path, data):

    with open(file_path, 'r+') as json_file:
        content = json.load(json_file)
        content.append(data)  
        json_file.seek(0)  #revenir au début du fichier
        json.dump(content, json_file)


EX = Namespace("http://example.org/entity/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
SCHEMA = Namespace("http://schema.org/")

def get_wikidata_neighbors(entity_id, limit=100):
    query = f"""
    SELECT ?p ?o WHERE {{
      wd:{entity_id} ?p ?o .
    }}
    LIMIT {limit}
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json",
             "User-Agent": "WikidataAgent/0.1 (krysto.dagues-de-la-hellerie@etu.univ-cotedazur.fr)"}
    response = requests.get(url, params={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"Erreur HTTP {response.status_code}: {response.text}")
        return []
    results = response.json()["results"]["bindings"]
    return [(r["p"]["value"], r["o"]["value"]) for r in results]

def get_entity_label(entity_id, lang="fr,en"):
    query = f"""
    SELECT ?label WHERE {{
      wd:{entity_id} rdfs:label ?label .
      FILTER (lang(?label) IN ({', '.join(f'"{l}"' for l in lang.split(','))}))
    }}
    LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json",
               "User-Agent": "WikidataAgent/0.1 (krysto.dagues-de-la-hellerie@etu.univ-cotedazur.fr)"}
    response = requests.get(url, params={"query": query}, headers=headers)
    
    if response.status_code != 200:
        print(f"Erreur HTTP {response.status_code} lors de la récupération du label")
        return None
    
    results = response.json()["results"]["bindings"]
    if results:
        return results[0]["label"]["value"]
    return None

def get_entity_labels_for_uris(uris):
    labels = {}
    for uri in uris:
        
        if uri.startswith("http://www.wikidata.org/entity/"):
            print(f"Récupération du label pour l'URI : {uri}")
            entity_id = uri.split("/")[-1]
            time.sleep(1)  #pause pour éviter erreru 429
            label = get_entity_label(entity_id)
            if label:
                labels[uri] = label
    
    return labels

def enrich_graph_with_neighbors(graph, entity_id, neighbors):
    subject = URIRef(f"http://www.wikidata.org/entity/{entity_id}")
    entities_to_label = set()  

    for p_str, o_str in neighbors:
        predicate = URIRef(p_str)
        
        if o_str.startswith("http://") or o_str.startswith("https://"):
            obj = URIRef(o_str)
            graph.add((obj, EX.isWikidataNeighborOf, subject))
            graph.add((obj, RDF.type, OWL.Thing))
            
            if o_str.startswith("http://www.wikidata.org/entity/"):
                entities_to_label.add(o_str)
        else:
            obj = Literal(o_str)
        
        graph.add((subject, predicate, obj))
    
    return entities_to_label

def add_labels_to_entities(graph, entity_uris):
    #ajoute skos preflabel
    labels = get_entity_labels_for_uris(entity_uris)
    for uri, label in labels.items():
        print(f"Ajout du label '{label}' pour l'entité {uri}")
        graph.add((URIRef(uri), SKOS.prefLabel, Literal(label)))
        print(f"Ajouté le label '{label}' pour l'entité {uri}")
    
    print(f"Ajouté {len(labels)} labels aux entités")

def add_wikidata_neighbors_to_graph(graph_path, output_path="enriched_graph.ttl", limit_per_entity=100):
   
    graph = rdflib.Graph()
    graph.parse(graph_path, format='turtle')
    
    graph.bind("ex", EX)
    graph.bind("skos", SKOS)
    graph.bind("wd", WD)
    graph.bind("wdt", WDT)
    graph.bind("schema", SCHEMA)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    
    query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?entity ?type WHERE {
        ?entity a ?type .
        FILTER(STRSTARTS(STR(?type), "https://www.wikidata.org/wiki/"))
    }
    """
    results = graph.query(query)
    
    print("Entités Wikidata trouvées :")
    all_entities_to_label = set()  
    
    for row in results:
        entity_uri = row.entity
        wikidata_uri = row.type
        wikidata_code = str(wikidata_uri).split("/")[-1]
        print(f"- {entity_uri} (wikidata: {wikidata_code})")
        
        time.sleep(1)  #pause pour eviter une erreur 429
        neighbors = get_wikidata_neighbors(wikidata_code, limit=limit_per_entity)
    
        entities_to_label = enrich_graph_with_neighbors(graph, wikidata_code, neighbors)
        all_entities_to_label.update(entities_to_label)
    
    print(f"Ajout de labels pour {len(all_entities_to_label)} entités...")
    add_labels_to_entities(graph, all_entities_to_label)
    
    
    graph.serialize(output_path, format="turtle")
    print(f"Graphe enrichi enregistré dans {output_path}")

def enrich_entity(graph, entity_id, limit=100):
    neighbors = get_wikidata_neighbors(entity_id, limit)
    entities_to_label = enrich_graph_with_neighbors(graph, entity_id, neighbors)
    add_labels_to_entities(graph, entities_to_label)
    return graph

# add_wikidata_neighbors_to_graph("finance/outputLinkerLinked.ttl")
# testlist = get_wikidata_neighbors("Q296782")
# print(testlist)
