import requests
import re
from difflib import SequenceMatcher
import rdflib
import os
import json
import time
from rdflib import Namespace, RDF, OWL, RDFS, URIRef, Literal,Graph
import statistics

def get_uri_wikidata(entite, langue="en"): # doit forcement retourner une URI wikidata # not used

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

def recherche_directe(terme, langue="fr"): #not used

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

def generer_transformations(entite): # not used

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
    entity_id = entity_url.split('/')[-1]  
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
    PREFIX ex: <http://example.org/>
    SELECT ?entity ?chunk ?label WHERE {
    {
    
        ?entity rel:mentionedIn ?chunk .
        ?chunk skos:prefLabel ?label .
        FILTER (CONTAINS(LCASE(str(?label)), LCASE(str(?entity_label))))
    }
    }
    
    """


    # query = """

    # PREFIX ex: <http://example.org/> 
    # PREFIX rel: <http://relations.example.org/> 
    # PREFIX skos: <http://www.w3.org/2004/02/skos/core#> 

    # SELECT ?entity ?chunk ?label WHERE {
    # {
    # ?entity rel:mentionedIn ?chunk .
    # ?chunk skos:prefLabel ?label .
    # FILTER (CONTAINS(LCASE(str(?label)), LCASE(str(?entity_label)))) }

    # UNION
    # {
    # ?entity skos:prefLabel ?label2 .
    # ?entity ex:isWikidataNeighborOf ?neighbor_entity .
    # ?neighbor a ?neighbor_entity .
    # ?neighbor rel:mentionedIn ?chunk .
    # OPTIONAL{?chunk skos:prefLabel ?label} . 
    # FILTER (CONTAINS(LCASE(str(?label2)), LCASE(str(?entity_label))) && bound(?label)) .
    # }
    # }
    # """

    results = graph.query(query, initBindings={'entity_label': entity})
    print(f"Nombre de résultats trouvés : {len(results)}")

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

    # Add neighboors chunks
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
    # add neighbor to the list if their label is not already in it
    for uri in mentioned_chunks:
        # get label of the neighbor chunk from the graph
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

        

    return l, chunk_already_Mentionned




def add_text_to_file(filepath, text) : # add text to a file without overwriting it
    with open(filepath, 'a') as file:
        file.write(text)



def initialize_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as json_file:
            json.dump([], json_file)  # empty list
    else:
        with open(file_path, 'w') as json_file:
            json.dump([], json_file) 

def add_to_json_file(file_path, data):

    with open(file_path, 'r+') as json_file:
        content = json.load(json_file)
        content.append(data)  
        json_file.seek(0)  #go at the begining of the file
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
            time.sleep(1)  #to avoid error 429
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

# PREFIX wd: <http://www.wikidata.org/entity/>
# PREFIX bd: <http://www.bigdata.com/rdf#>
# PREFIX wikibase: <http://wikiba.se/ontology#>

# SELECT ?predicate ?proplabel ?object ?objectLabel
# WHERE {
# SERVICE <https://query.wikidata.org/sparql> {

#     SELECT ?predicate ?proplabel ?object ?objectLabel {
#         wd:Q93183876 ?predicate ?object .
#         FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
#         ?object rdfs:label ?objectLabel .
#         FILTER(LANG(?objectLabel) = "en" || LANG(?objectLabel) = "fr")

#         SERVICE wikibase:label { bd:serviceParam wikibase:language "en". bd:serviceParam wikibase:language "fr". } 
#         ?prop wikibase:directClaim ?predicate . 
#         ?prop rdfs:label ?proplabel .
#         FILTER(LANG(?proplabel) = "en" || LANG(?proplabel) = "fr" )
#     }
# }
# }
# LIMIT 500

# def add_wikidata_neighbors_to_graph(graph_path, output_path="enriched_graph.ttl", limit_per_entity=100):

#     import rdflib
#     from rdflib import URIRef, Literal, Graph
#     from rdflib.namespace import RDF, RDFS, OWL, SKOS
#     import time
    
#     WD = rdflib.Namespace("http://www.wikidata.org/wiki/")
#     WDT = rdflib.Namespace("http://www.wikidata.org/prop/direct/")
#     SCHEMA = rdflib.Namespace("http://schema.org/")
#     EX = rdflib.Namespace("http://example.org/")
    
#     graph = Graph()
#     graph.parse(graph_path, format='turtle')
    
#     graph.bind("ex", EX)
#     graph.bind("skos", SKOS)
#     graph.bind("wd", WD)
#     graph.bind("wdt", WDT)
#     graph.bind("schema", SCHEMA)
#     graph.bind("owl", OWL)
#     graph.bind("rdfs", RDFS)

#     labelList = []
    
#     #take wikidata entities in the graph
#     query = """
#     PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#     SELECT ?entity ?type
#     WHERE {
#         ?entity a ?type .
#         FILTER(STRSTARTS(STR(?type), "https://www.wikidata.org/wiki/"))
#     }
#     """
#     results = graph.query(query)
    
#     print(f"{len(results)} entités Wikidata")
    
#     wikidata_endpoint = "https://query.wikidata.org/sparql"
    
#     #For each entity seach neighbors with label and add them to the graph
#     for row in results:
#         entity_uri = row.entity
#         wikidata_uri = row.type
#         wikidata_id = str(wikidata_uri).split("/")[-1]
        
#         print(f"entité: {entity_uri} (Wikidata: {wikidata_id})")
        
#         # outgoing relations
#         sparql_query_out = f"""
#         PREFIX wd: <http://www.wikidata.org/entity/>
#             SELECT ?predicate ?predicateLabel ?object ?objectLabel
#                 WHERE {{
#                 SERVICE <https://query.wikidata.org/sparql> {{
#                     wd:{wikidata_id} ?predicate ?object .
#                     FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
#                     ?object rdfs:label ?objectLabel .
#                     FILTER(LANG(?objectLabel) = "fr" || LANG(?objectLabel) = "en")
#                     ?predicate rdfs:label ?predicateLabel .
#                     FILTER(LANG(?objectLabel) = "fr" || LANG(?objectLabel) = "en")
#                 }}
#                 }}
#                 LIMIT 500
#         """
        
#         # incoming relations
#         sparql_query_in = f"""
#         PREFIX wd: <http://www.wikidata.org/entity/>
#         SELECT ?predicate ?subject ?subjectLabel
#         WHERE {{
#           SERVICE <https://query.wikidata.org/sparql> {{
#             ?subject ?predicate wd:{wikidata_id} .
#             FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
#             ?subject rdfs:label ?subjectLabel .
#             FILTER(LANG(?subjectLabel) = "fr" || LANG(?subjectLabel) = "en")
#           }}
#         }}
#         LIMIT 500
#         """
        
#         try:
#             results_out = graph.query(sparql_query_out)
#             print(f"RESULTS OUT: {results_out}, TAILLE : {len(results_out)}")
#             results_in = graph.query(sparql_query_in)
#             print(f"RESULTS IN: {results_in}, TAILLE : {len(results_in)}")
            
#             total_results = len(results_out) + len(results_in)
#             print(f"{total_results} relations trouvées")
            
#             entity_count = 0
            
#             for result in results_out:

               
#                 predicate_uri = str(result.predicate)
               
               
#                 object_value = str(result.object)
                
             

#                 clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
#                 clean_predicate = clean_predicate.replace("direct/", "")
#                 clean_predicate_uri = WDT[clean_predicate]
                
#                 if isinstance(result.object, URIRef):
#                     obj = URIRef(object_value)
                     
#                     graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, obj))
                    
#                     graph.add((obj, RDF.type, OWL.Thing))
                    
#                     #add isWikidataNeighborOf relation used when we want to get chunks linked to the entity
#                     graph.add((obj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
#                     graph.add((obj, EX.relationDirection, Literal("outgoing")))
                    
#                     #add label
#                     label = str(result.objectLabel)
#                     graph.add((obj, SKOS.prefLabel, Literal(label)))
#                     if(label not in labelList):
#                         labelList.append(label)
#                     print(f"Ajouté le label '{label}' pour l'entité {obj}")
#                     entity_count += 1
#                 else:
#                     print("!!!===================ALERT LITTERAL VALUE =========================================")
#                     print(Literal(object_value))
#                     #if it is a litteral value, add it directly
#                     graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, Literal(object_value)))
            
#             for result in results_in:
#                 subject_uri = str(result.subject)
#                 predicate_uri = str(result.predicate)
#                 subject_label = str(result.subjectLabel)
#                 if(label not in labelList):
#                     labelList.append(subject_label)
#                 print(f"Ajouté le label '{subject_label}' pour l'entité {subject_uri}")
                
#                 clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
#                 clean_predicate = clean_predicate.replace("direct/", "")
#                 clean_predicate_uri = WDT[clean_predicate]
                
#                 subj = URIRef(subject_uri)
#                 graph.add((subj, clean_predicate_uri, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                
#                 graph.add((subj, RDF.type, OWL.Thing))
                
#                 graph.add((subj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
#                 graph.add((subj, EX.relationDirection, Literal("incoming")))
                
#                 graph.add((subj, SKOS.prefLabel, Literal(subject_label)))
#                 entity_count += 1
            
#             print(f"{entity_count} entités ajoutées")
            
            
#             time.sleep(1)  #to avoid error 429 
#         except Exception as e:
#             print(f"Erreur lors de la requête SPARQL pour {wikidata_id}: {e} \n")
           
    
#     graph.serialize(output_path, format="turtle")
#     print(f"Graphe enrichi enregistré dans {output_path}")
    
#     return graph, labelList

# def add_wikidata_neighbors_to_graph(graph_path, output_path="enriched_graph.ttl", limit_per_entity=100):

#     import rdflib
#     from rdflib import URIRef, Literal, Graph
#     from rdflib.namespace import RDF, RDFS, OWL, SKOS
#     import time
    
#     WD = rdflib.Namespace("http://www.wikidata.org/wiki/")
#     WDT = rdflib.Namespace("http://www.wikidata.org/prop/direct/")
#     SCHEMA = rdflib.Namespace("http://schema.org/")
#     EX = rdflib.Namespace("http://example.org/")
    
#     graph = Graph()
#     graph.parse(graph_path, format='turtle')
    
#     graph.bind("ex", EX)
#     graph.bind("skos", SKOS)
#     graph.bind("wd", WD)
#     graph.bind("wdt", WDT)
#     graph.bind("schema", SCHEMA)
#     graph.bind("owl", OWL)
#     graph.bind("rdfs", RDFS)

#     labelList = []
    
#     #take wikidata entities in the graph
#     query = """
#     PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#     SELECT ?entity ?type
#     WHERE {
#         ?entity a ?type .
#         FILTER(STRSTARTS(STR(?type), "https://www.wikidata.org/wiki/"))
#     }
#     """
#     results = graph.query(query)
    
#     print(f"{len(results)} entités Wikidata")
    
#     wikidata_endpoint = "https://query.wikidata.org/sparql"
    
#     #For each entity search neighbors with label and add them to the graph
#     for row in results:
#         entity_uri = row.entity
#         wikidata_uri = row.type
#         wikidata_id = str(wikidata_uri).split("/")[-1]
        
#         print(f"entité: {entity_uri} (Wikidata: {wikidata_id})")
        
#         # outgoing relations with property labels
#         sparql_query_out = f"""
#         PREFIX wd: <http://www.wikidata.org/entity/>
#         PREFIX bd: <http://www.bigdata.com/rdf#>
#         PREFIX wikibase: <http://wikiba.se/ontology#>
#         SELECT ?predicate ?proplabel ?object ?objectLabel
#         WHERE {{
#         SERVICE <https://query.wikidata.org/sparql> {{
#             SELECT ?predicate ?proplabel ?object ?objectLabel {{
#                 wd:{wikidata_id} ?predicate ?object .
#                 FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
#                 ?object rdfs:label ?objectLabel .
#                 FILTER(LANG(?objectLabel) = "en" || LANG(?objectLabel) = "fr")
#                 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". bd:serviceParam wikibase:language "fr". }}
#                 ?prop wikibase:directClaim ?predicate .
#                 ?prop rdfs:label ?proplabel .
#                 FILTER(LANG(?proplabel) = "en" || LANG(?proplabel) = "fr" )
#             }}
#         }}
#         }}
#         LIMIT 500
#         """
        
#         # incoming relations with property labels
#         sparql_query_in = f"""
#         PREFIX wd: <http://www.wikidata.org/entity/>
#         PREFIX bd: <http://www.bigdata.com/rdf#>
#         PREFIX wikibase: <http://wikiba.se/ontology#>
#         SELECT ?predicate ?proplabel ?subject ?subjectLabel
#         WHERE {{
#           SERVICE <https://query.wikidata.org/sparql> {{
#             SELECT ?predicate ?proplabel ?subject ?subjectLabel {{
#                 ?subject ?predicate wd:{wikidata_id} .
#                 FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
#                 ?subject rdfs:label ?subjectLabel .
#                 FILTER(LANG(?subjectLabel) = "fr" || LANG(?subjectLabel) = "en")
#                 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". bd:serviceParam wikibase:language "fr". }}
#                 ?prop wikibase:directClaim ?predicate .
#                 ?prop rdfs:label ?proplabel .
#                 FILTER(LANG(?proplabel) = "en" || LANG(?proplabel) = "fr" )
#             }}
#           }}
#         }}
#         LIMIT 500
#         """
        
#         try:
#             results_out = graph.query(sparql_query_out)
#             print(f"RESULTS OUT: {results_out}, TAILLE : {len(results_out)}")
#             results_in = graph.query(sparql_query_in)
#             print(f"RESULTS IN: {results_in}, TAILLE : {len(results_in)}")
            
#             total_results = len(results_out) + len(results_in)
#             print(f"{total_results} relations trouvées")
            
#             entity_count = 0
            
#             for result in results_out:
#                 predicate_uri = str(result.predicate)
#                 object_value = str(result.object)
#                 prop_label = str(result.proplabel)

#                 clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
#                 clean_predicate = clean_predicate.replace("direct/", "")
#                 clean_predicate_uri = WDT[clean_predicate]
                
#                 # Add property label to the graph
#                 graph.add((clean_predicate_uri, RDFS.label, Literal(prop_label)))
                
#                 if isinstance(result.object, URIRef):
#                     obj = URIRef(object_value)
                     
#                     graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, obj))
                    
#                     graph.add((obj, RDF.type, OWL.Thing))
                    
#                     #add isWikidataNeighborOf relation used when we want to get chunks linked to the entity
#                     graph.add((obj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
#                     graph.add((obj, EX.relationDirection, Literal("outgoing")))
                    
#                     #add label
#                     label = str(result.objectLabel)
#                     graph.add((obj, SKOS.prefLabel, Literal(label)))
#                     if(label not in labelList):
#                         labelList.append(label)
#                     print(f"Ajouté le label '{label}' pour l'entité {obj}")
#                     print(f"Ajouté le label de propriété '{prop_label}' pour {clean_predicate_uri}")
#                     entity_count += 1
#                 else:
#                     print("!!!===================ALERT LITTERAL VALUE =========================================")
#                     print(Literal(object_value))
#                     #if it is a literal value, add it directly
#                     graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, Literal(object_value)))
            
#             for result in results_in:
#                 subject_uri = str(result.subject)
#                 predicate_uri = str(result.predicate)
#                 subject_label = str(result.subjectLabel)
#                 prop_label = str(result.proplabel)
                
#                 if(subject_label not in labelList):
#                     labelList.append(subject_label)
#                 print(f"Ajouté le label '{subject_label}' pour l'entité {subject_uri}")
                
#                 clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
#                 clean_predicate = clean_predicate.replace("direct/", "")
#                 clean_predicate_uri = WDT[clean_predicate]
                
#                 # Add property label to the graph
#                 graph.add((clean_predicate_uri, RDFS.label, Literal(prop_label)))
                
#                 subj = URIRef(subject_uri)
#                 graph.add((subj, clean_predicate_uri, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                
#                 graph.add((subj, RDF.type, OWL.Thing))
                
#                 graph.add((subj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
#                 graph.add((subj, EX.relationDirection, Literal("incoming")))
                
#                 graph.add((subj, SKOS.prefLabel, Literal(subject_label)))
#                 print(f"Ajouté le label de propriété '{prop_label}' pour {clean_predicate_uri}")
#                 entity_count += 1
            
#             print(f"{entity_count} entités ajoutées")
            
#             time.sleep(1)  #to avoid error 429 
#         except Exception as e:
#             print(f"Erreur lors de la requête SPARQL pour {wikidata_id}: {e} \n")
           
    
#     graph.serialize(output_path, format="turtle")
#     print(f"Graphe enrichi enregistré dans {output_path}")
    
#     return graph, labelList



    #============================================================================================================

def add_wikidata_neighbors_to_graph(graph_path, output_path="enriched_graph.ttl", limit_per_entity=100):

    import rdflib
    from rdflib import URIRef, Literal, Graph
    from rdflib.namespace import RDF, RDFS, OWL, SKOS
    import time
    import requests
    
    WD = rdflib.Namespace("http://www.wikidata.org/wiki/")
    WDT = rdflib.Namespace("http://www.wikidata.org/prop/direct/")
    SCHEMA = rdflib.Namespace("http://schema.org/")
    EX = rdflib.Namespace("http://example.org/")
    
    graph = Graph()
    graph.parse(graph_path, format='turtle')
    
    graph.bind("ex", EX)
    graph.bind("skos", SKOS)
    graph.bind("wd", WD)
    graph.bind("wdt", WDT)
    graph.bind("schema", SCHEMA)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)

    labelList = []
    
    #take wikidata entities in the graph
    query = """
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?entity ?type
    WHERE {
        ?entity a ?type .
        FILTER(STRSTARTS(STR(?type), "https://www.wikidata.org/wiki/"))
    }
    """
    results = graph.query(query)
    
    print(f"{len(results)} entités Wikidata")
    
    wikidata_endpoint = "https://query.wikidata.org/sparql"
    
    def execute_sparql_query(query, endpoint):
        """Exécute une requête SPARQL via HTTP"""
        headers = {
            'User-Agent': 'Python SPARQL Client',
            'Accept': 'application/sparql-results+json'
        }
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    
    #For each entity search neighbors with label and add them to the graph
    for row in results:
        entity_uri = row.entity
        wikidata_uri = row.type
        wikidata_id = str(wikidata_uri).split("/")[-1]
        
        print(f"entité: {entity_uri} (Wikidata: {wikidata_id})")
        
        # outgoing relations with property labels
        sparql_query_out = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        SELECT ?predicate ?proplabel ?object ?objectLabel
        WHERE {{
        SERVICE <https://query.wikidata.org/sparql> {{
            SELECT ?predicate ?proplabel ?object ?objectLabel {{
                wd:{wikidata_id} ?predicate ?object .
                FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
                ?object rdfs:label ?objectLabel .
                FILTER(LANG(?objectLabel) = "en" || LANG(?objectLabel) = "fr")
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". bd:serviceParam wikibase:language "fr". }}
                ?prop wikibase:directClaim ?predicate .
                ?prop rdfs:label ?proplabel .
                FILTER(LANG(?proplabel) = "en" || LANG(?proplabel) = "fr" )
            }}
        }}
        }}
        LIMIT 500
        """
        
        # incoming relations with property labels
        sparql_query_in = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        SELECT ?predicate ?proplabel ?subject ?subjectLabel
        WHERE {{
          SERVICE <https://query.wikidata.org/sparql> {{
            SELECT ?predicate ?proplabel ?subject ?subjectLabel {{
                ?subject ?predicate wd:{wikidata_id} .
                FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
                ?subject rdfs:label ?subjectLabel .
                FILTER(LANG(?subjectLabel) = "fr" || LANG(?subjectLabel) = "en")
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". bd:serviceParam wikibase:language "fr". }}
                ?prop wikibase:directClaim ?predicate .
                ?prop rdfs:label ?proplabel .
                FILTER(LANG(?proplabel) = "en" || LANG(?proplabel) = "fr" )
            }}
          }}
        }}
        LIMIT 500
        """
        
        try:
            # Exécuter les requêtes via HTTP
            results_out_data = execute_sparql_query(sparql_query_out, wikidata_endpoint)
            results_in_data = execute_sparql_query(sparql_query_in, wikidata_endpoint)
            
            results_out = results_out_data['results']['bindings']
            results_in = results_in_data['results']['bindings']
            
            print(f"RESULTS OUT: {len(results_out)} résultats")
            print(f"RESULTS IN: {len(results_in)} résultats")
            
            total_results = len(results_out) + len(results_in)
            print(f"{total_results} relations trouvées")
            
            entity_count = 0
            
            for result in results_out:
                predicate_uri = result['predicate']['value']
                object_value = result['object']['value']
                prop_label = result['proplabel']['value']

                clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
                clean_predicate = clean_predicate.replace("direct/", "")
                clean_predicate_uri = WDT[clean_predicate]
                
                # Add property label to the graph
                graph.add((clean_predicate_uri, RDFS.label, Literal(prop_label)))
                
                if result['object']['type'] == 'uri':
                    obj = URIRef(object_value)
                     
                    graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, obj))
                    
                    graph.add((obj, RDF.type, OWL.Thing))
                    
                    #add isWikidataNeighborOf relation used when we want to get chunks linked to the entity
                    graph.add((obj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                    graph.add((obj, EX.relationDirection, Literal("outgoing")))
                    
                    #add label
                    label = result['objectLabel']['value']
                    graph.add((obj, SKOS.prefLabel, Literal(label)))
                    if(label not in labelList):
                        labelList.append(label)
                    print(f"Ajouté le label '{label}' pour l'entité {obj}")
                    print(f"Ajouté le label de propriété '{prop_label}' pour {clean_predicate_uri}")
                    entity_count += 1
                else:
                    print("!!!===================ALERT LITTERAL VALUE =========================================")
                    print(Literal(object_value))
                    #if it is a literal value, add it directly
                    graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, Literal(object_value)))
            
            for result in results_in:
                subject_uri = result['subject']['value']
                predicate_uri = result['predicate']['value']
                subject_label = result['subjectLabel']['value']
                prop_label = result['proplabel']['value']
                
                if(subject_label not in labelList):
                    labelList.append(subject_label)
                print(f"Ajouté le label '{subject_label}' pour l'entité {subject_uri}")
                
                clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
                clean_predicate = clean_predicate.replace("direct/", "")
                clean_predicate_uri = WDT[clean_predicate]
                
                # Add property label to the graph
                graph.add((clean_predicate_uri, RDFS.label, Literal(prop_label)))
                
                subj = URIRef(subject_uri)
                graph.add((subj, clean_predicate_uri, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                
                graph.add((subj, RDF.type, OWL.Thing))
                
                graph.add((subj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                graph.add((subj, EX.relationDirection, Literal("incoming")))
                
                graph.add((subj, SKOS.prefLabel, Literal(subject_label)))
                print(f"Ajouté le label de propriété '{prop_label}' pour {clean_predicate_uri}")
                entity_count += 1
            
            print(f"{entity_count} entités ajoutées")
            
            time.sleep(1)  #to avoid error 429 
        except Exception as e:
            print(f"Erreur lors de la requête SPARQL pour {wikidata_id}: {e} \n")
           
    
    graph.serialize(output_path, format="turtle")
    print(f"Graphe enrichi enregistré dans {output_path}")
    
    return graph, labelList



def enrich_entity(graph, entity_id, limit=100):
    neighbors = get_wikidata_neighbors(entity_id, limit)
    entities_to_label = enrich_graph_with_neighbors(graph, entity_id, neighbors)
    add_labels_to_entities(graph, entities_to_label)
    return graph


def make_property_stats(graph_path, output_path="property_stats.json"):
    import rdflib
    import json
    
    # count properties and their occurrences in the graph
    graph = rdflib.Graph()
    graph.parse(graph_path, format='turtle')
    property_stats = {}
    
    query = """
    SELECT ?property (COUNT(DISTINCT ?subject) AS ?count)
    WHERE {
        ?subject ?property ?object .
    }
    GROUP BY ?property
    """
    
    results = graph.query(query)
    for row in results:
        property_uri = str(row.property)
        
        count_value = row['count']  
        
        if hasattr(count_value, 'toPython'):
            count = int(count_value.toPython())
        else:
            count = int(count_value)
        
        property_stats[property_uri] = count
 
    with open(output_path, 'w') as f:
        json.dump(property_stats, f, indent=4)
    
    return property_stats

def filter_neigbor(graph_path, property_stats_path, output_path="filtered_graph.ttl", threshold=20): # do not use
    # for each entity in the graph that as the iswikidataneighborof relation, see if it as a relation wich occurs less than threshold times
    # if it is NOT the case, we remove the relation from the graph
    import rdflib
    import json
    graph = rdflib.Graph()
    graph.parse(graph_path, format='turtle')
    with open(property_stats_path, 'r') as f:
        property_stats = json.load(f)
    neighborlist = []
    for s,p,o in graph.triples((None, rdflib.URIRef("http://example.org/isWikidataNeighborOf"), None)):
        if s:
            neighborlist.append(s)
            print("Neighbor accepted :", s)
    print(f"Nombre de voisins trouvés : {len(neighborlist)}")
    print(f"debut neghborlist : {neighborlist[:10]}")

    #then for each neighbor, check if it has a relation that occurs less than threshold times, if not remove from list and graph
    reslist = []
    for neighbor in neighborlist:
        print(f"Vérification du voisin : {neighbor}")
        for p in graph.predicates(subject=neighbor):
            if str(p) in property_stats and property_stats[str(p)] <= threshold:
                print(f" voisin {neighbor} validé.")
                if neighbor not in reslist:
                    reslist.append(neighbor)
                
    #delete all subject from the graph that are not in reslist but in neighborlist
    for s in neighborlist:
        if s not in reslist:
            print(f"Suppression du voisin {s} du graphe")
            graph.remove((s, None, None))  # remove all triples with this subject
            graph.remove((None, None, s))  # remove all triples with this object      
                
    
    graph.serialize(output_path, format='turtle')
    print(f"Graphe filtré enregistré dans {output_path}")

    labelList = []
    for entity in reslist:
        query = """
        SELECT ?label WHERE {
            <""" + str(entity) + """> skos:prefLabel ?label .
        }
        """
        results = graph.query(query)
        for row in results:
            label = str(row.label)
            if label not in labelList:
                labelList.append(label)
                print(f"Ajouté le label '{label}' pour l'entité {entity}")


    return labelList # to add in the DAO later 

def calculate_quantiles_on_property_stats(property_stats_path):
    # calculate the median of the property stats
    with open(property_stats_path, 'r') as f:
        property_stats = json.load(f)

    counts = list(property_stats.values())
    if not counts:
        return None

    quantile = statistics.quantiles(counts, n=4)
    print(f"Quartiles des occurrences des propriétés : {quantile}")
    median = statistics.median(counts)
    print(f"Médiane des occurrences des propriétés : {median}")
    quantile_75 = quantile[2]  
    print(f"Quantile 75 des occurrences des propriétés : {quantile_75}")
    return quantile_75


def get_wikidata_info(wikidata_uri):
    # exctract 'ID Qxxxx
    match = re.search(r'Q\d+', str(wikidata_uri))
    if not match:
        return {"label": None, "description": None, "aliases": []}

    qid = match.group(0)

   
    url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?label ?description ?alias WHERE {{
      wd:{qid} rdfs:label ?label .
      FILTER (lang(?label) = "en")
      OPTIONAL {{
        wd:{qid} schema:description ?description .
        FILTER (lang(?description) = "en")
      }}
      OPTIONAL {{
        wd:{qid} skos:altLabel ?alias .
        FILTER (lang(?alias) = "en")
      }}
    }}
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": query}, headers=headers)
    if response.status_code != 200:
        return {"label": None, "description": None, "aliases": []}

    results = response.json()["results"]["bindings"]

    label = None
    description = None
    aliases = []

    for result in results:
        if 'label' in result:
            label = result['label']['value']
        if 'description' in result:
            description = result['description']['value']
        if 'alias' in result:
            aliases.append(result['alias']['value'])

    # max 200 characters for description
    if description and len(description) > 200:
        description = description[:200] + "..."

    return {"label": label, "description": description, "aliases": list(set(aliases))}


def get_entity_info(graph_path, label):
    g = rdflib.Graph()
    g.parse(graph_path, format="turtle")

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    for s, _, o in g.triples((None, SKOS.prefLabel, None)):
        if str(o) == label:
            for _, _, t in g.triples((s, RDF.type, None)):
                t_str = str(t)
                if "wikidata.org" in t_str:
                    info = get_wikidata_info(t)
                    type_label = info['label'] if info['label'] else t_str
                    desc = info['description'] if info['description'] else "No description"
                    aliases = ", ".join(info['aliases']) if info['aliases'] else "No aliases"
                    return (
                        f"{label} ({s}) is of type {type_label} ({t_str})\n"
                        f"Description: {desc}\n"
                        f"Also known as: {aliases}"
                    )
                else:
                    return f"{label} ({s}) is of type {t_str} (non-Wikidata type)"

            return f"{label} ({s}) has no rdf:type"

    return f"No entity found with label '{label}'"


def analyze_entity(graph_path: str, label: str, ontology_path: str) -> str:
    # Load the main graph
    g = Graph()
    try:
        g.parse(graph_path, format="turtle")
    except Exception as e:
        return f"Error loading graph: {e}"
    
    # Load the ontology
    ontology = Graph()
    try:
        ontology.parse(ontology_path, format="turtle")
    except Exception as e:
        print(f"Warning: Error loading ontology: {e}")
    
    # Find entity URI by label (case-sensitive first, then case-insensitive)
    entity_uri = None
    for s, p, o in g.triples((None, SKOS.prefLabel, Literal(label))):
        entity_uri = s
        break
    if not entity_uri:
        for s, p, o in g.triples((None, RDFS.label, Literal(label))):
            entity_uri = s
            break
    
    # If not found, try case-insensitive search
    if not entity_uri:
        for s, p, o in g.triples((None, SKOS.prefLabel, None)):
            if str(o).lower() == label.lower():
                entity_uri = s
                break
    if not entity_uri:
        for s, p, o in g.triples((None, RDFS.label, None)):
            if str(o).lower() == label.lower():
                entity_uri = s
                break
    
    if not entity_uri:
        # Debug: voir toutes les entités avec leurs labels
        print(f"Debug: Searching for label '{label}'")
        print("Available entities:")
        for s, p, o in g.triples((None, SKOS.prefLabel, None)):
            print(f"  {s} -> {o}")
        for s, p, o in g.triples((None, RDFS.label, None)):
            print(f"  {s} -> {o}")
        return f"Entity '{label}' has no complementary information."
    
    print(f"Debug: Found entity URI: {entity_uri}")
    
    # Check if the entity URI itself is a Wikidata entity
    wikidata_id = None
    entity_type = None
    if str(entity_uri).startswith("http://www.wikidata.org/entity/Q"):
        wikidata_id = str(entity_uri).split("/")[-1]
        print(f"Debug: Entity URI is directly Wikidata: {wikidata_id}")
    else:
        # Get entity type
        entity_type = None
        for s, p, o in g.triples((entity_uri, RDF.type, None)):
            entity_type = o
            print(f"Debug: Found entity type: {entity_type}")
            break
        
        # Check if it's a Wikidata entity (via rdf:type)
        if entity_type and str(entity_type).startswith("https://www.wikidata.org/wiki/Q"):
            wikidata_id = str(entity_type).split("/")[-1]
            print(f"Debug: Found Wikidata ID from type: {wikidata_id}")
        else:
            print("Debug: No Wikidata type found, checking for direct links...")
            # Check for direct Wikidata links via owl:sameAs or other properties
            for s, p, o in g.triples((entity_uri, None, None)):
                if str(o).startswith("https://www.wikidata.org/wiki/Q"):
                    wikidata_id = str(o).split("/")[-1]
                    print(f"Debug: Found Wikidata ID from property {p}: {wikidata_id}")
                    break
    
    print(f"Debug: Final Wikidata ID: {wikidata_id}")
    
    # query Wikidata
    if wikidata_id:
        
        
        sparql_query = f"""
        SELECT ?typeLabel ?description ?altLabel WHERE {{
          wd:{wikidata_id} schema:description ?description .
          OPTIONAL {{ 
            wd:{wikidata_id} wdt:P31 ?type .
            SERVICE wikibase:label {{ 
              bd:serviceParam wikibase:language "en" .
              ?type rdfs:label ?typeLabel .
            }}
          }}
          OPTIONAL {{ wd:{wikidata_id} skos:altLabel ?altLabel . }}
          
          FILTER(LANG(?description) = "en")
          FILTER(LANG(?altLabel) = "en" || !BOUND(?altLabel))
        }}
        LIMIT 10
        """
        
        try:
            endpoint = "https://query.wikidata.org/sparql"
            headers = {
                'User-Agent': 'RDF-Entity-Analyzer/1.0',
                'Accept': 'application/json'
            }
            
            
            response = requests.get(
                endpoint,
                params={'query': sparql_query, 'format': 'json'},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                
                if bindings:
                    result = bindings[0]
                    
                    
                    type_label = result.get('typeLabel', {}).get('value', None)
                    description = result.get('description', {}).get('value', '')
                    alt_labels = [b.get('altLabel', {}).get('value', '') 
                                 for b in bindings if b.get('altLabel')]
                    
                    # Limit description to 200 characters
                    if len(description) > 200:
                        description = description[:199] + "..."
                    
                    # Limit aliases to 200 characters total
                    alt_labels_str = ", ".join(alt_labels) if alt_labels else "No alternative names"
                    if len(alt_labels_str) > 200:
                        alt_labels_str = alt_labels_str[:199] + "..."
                    
                    
                    if type_label:
                        return f"{label} is of type {type_label}, has description: {description}, is also known as: {alt_labels_str}"
                    else:
                        return f"{label} (type unknown), has description: {description}, is also known as: {alt_labels_str}"
                    
        except Exception as e:
            pass
    
    # Check if it's an ontology entity (if we have an entity_type)
    if entity_type:
        for s, p, o in ontology.triples((entity_type, None, None)):
            # Found in ontology, get label
            for s2, p2, o2 in ontology.triples((entity_type, SKOS.prefLabel, None)):
                return f"{label} is of type {o2}"
            for s2, p2, o2 in ontology.triples((entity_type, RDFS.label, None)):
                return f"{label} is of type {o2}"
            # If no label found, use URI
            return f"{label} is of type {str(entity_type)}"
    
    return f"Entity '{label}' has no complementary information."


import requests
import time
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, SKOS
import difflib

def align_unlinked_entities_to_wikidata(ttl_file_path: str, output_path: str = None):
    #try to align with wikidata entities with rel:mentioned if not aligned with either ontology or wikidata

    g = Graph()
    g.parse(ttl_file_path, format='turtle')

    REL = Namespace("http://example.org/rel/")

    query = """
    SELECT ?entity ?label WHERE {
        ?entity skos:prefLabel ?label .
        ?entity rel:mentionedIn ?chunk .
        FILTER NOT EXISTS { ?entity rdf:type ?type }
    }
    """
    
    g.bind("rel", REL)
    g.bind("skos", SKOS)
    g.bind("rdf", RDF)
    
    results = g.query(query)
    entities_to_align = [(row.entity, str(row.label)) for row in results]
    
    print(f"Trouvé {len(entities_to_align)} entités à aligner")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'WikidataAligner/1.0'
    })
    
    aligned_count = 0
    
    for entity_uri, entity_label in entities_to_align:
        print(f"Traitement: {entity_label}")

        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            'action': 'wbsearchentities',
            'search': entity_label,
            'language': 'en',
            'format': 'json',
            'limit': 5,
            'type': 'item'
        }
        
        try:
            response = session.get(search_url, params=search_params)
            response.raise_for_status()
            data = response.json()
            
            if 'search' in data and data['search']:
                candidates = data['search']

                best_match = None
                best_score = 0.0
                
                for candidate in candidates:
                    candidate_label = candidate.get('label', '')
                    candidate_description = candidate.get('description', '')

                    score = difflib.SequenceMatcher(None, 
                                                  entity_label.lower(), 
                                                  candidate_label.lower()).ratio()

                    if entity_label.lower() == candidate_label.lower():
                        score = 1.0
                    
                    if score > best_score and score >= 0.5:
                        best_score = score
                        best_match = candidate
                
                if best_match:
                    wikidata_id = best_match['id']
                    wikidata_uri = URIRef(f"https://www.wikidata.org/wiki/{wikidata_id}")
                    description = best_match.get('description', '')
                    
                    g.add((entity_uri, RDF.type, wikidata_uri))
                    g.add((entity_uri, REL.alignedWith, wikidata_uri))
                    g.add((entity_uri, REL.alignmentScore, Literal(best_score)))
                    
                    if description:
                        g.add((entity_uri, RDFS.comment, Literal(description)))
                    
                    print(f"  -> aligned with {wikidata_id} (score: {best_score:.3f})")
                    aligned_count += 1
                else:
                    print(f"  -> no sufficient correspondence")
            else:
                print(f"  -> no results found ")
                
        except requests.RequestException as e:
            print(f"  -> Error API: {e}")
        
       
        time.sleep(0.1)
    
    if output_path is None:
        base_name = ttl_file_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_aligned.ttl"
    
    g.serialize(destination=output_path, format='turtle')
    print(f"Alignement terminé: {aligned_count} entités alignées")
    print(f"Fichier sauvegardé: {output_path}")
    
    return output_path


import rdflib
from rdflib import Graph, Namespace, URIRef
import requests
import time
from typing import Dict, List


import time
import requests
from rdflib import Graph, Namespace
from typing import List, Dict, Set
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_property_labels_cache = {}
_cache_lock = threading.Lock()
_cache_file = 'property_cache.pkl'

def load_cache():
    global _property_labels_cache
    try:
        if os.path.exists(_cache_file):
            with open(_cache_file, 'rb') as f:
                _property_labels_cache = pickle.load(f)
            print(f"Cache loaded: {len(_property_labels_cache)} properties")
        else:
            _property_labels_cache = {}
    except Exception as e:
        print(f"Error loading cache: {e}")
        _property_labels_cache = {}

def save_cache():
    try:
        with open(_cache_file, 'wb') as f:
            pickle.dump(_property_labels_cache, f)
        print(f"Cache saved: {len(_property_labels_cache)} properties")
    except Exception as e:
        print(f"Error saving cache: {e}")

def preload_common_properties():
    common_properties = [
        'P31',   # instance of
        'P279',  # subclass of
        'P106',  # occupation
        'P19',   # place of birth
        'P20',   # place of death
        'P27',   # country of citizenship
        'P21',   # sex or gender
        'P569',  # date of birth
        'P570',  # date of death
        'P17',   # country
        'P131',  # located in
        'P159',  # headquarters location
        'P36',   # capital
        'P1376', # capital of
        'P150',  # contains
        'P361',  # part of
        'P527',  # has part
        'P276',  # location
        'P138',  # named after
        'P1435', # heritage designation
        'P625',  # coordinate location
        'P580',  # start time
        'P582',  # end time
        'P585',  # point in time
        'P571',  # inception
        'P577',  # publication date
        'P1001', # applies to jurisdiction
        'P1269', # facet of
        'P910',  # topic's main category
        'P373',  # Commons category
        'P856',  # official website
        'P1448', # official name
        'P1705', # native label
        'P18',   # image
        'P154',  # logo image
        'P94',   # coat of arms
        'P41',   # flag image
        'P242',  # locator map image
        'P2633', # geoshape
        'P1313', # office held by head of government
        'P1906', # office held by head of state
        'P6',    # head of government
        'P35',   # head of state
        'P123',  # publisher
        'P50',   # author
        'P2860', # cites
        'P921',  # main subject
        'P3373', # sibling
        'P22',   # father
        'P25',   # mother
        'P26',   # spouse
        'P40',   # child
        'P39',   # position held
        'P69',   # educated at
        'P108',  # employer
        'P463',  # member of
        'P1344', # participant in
        'P1066', # student of
        'P802',  # student
        'P937',  # work location
        'P551',  # residence
        'P800',  # notable work
        'P1412', # languages spoken
        'P103',  # native language
        'P1303', # instrument
        'P101',  # field of work
        'P136',  # genre
        'P641',  # sport
        'P413',  # position played
        'P54',   # member of sports team
        'P1532', # country for sport
        'P495',  # country of origin
        'P364',  # original language
        'P407',  # language of work
        'P710',  # participant
        'P1923', # participating team
        'P179',  # part of the series
        'P155',  # follows
        'P156',  # followed by
        'P144',  # based on
        'P1889', # different from
        'P460',  # said to be the same as
        'P1830', # owner of
        'P127',  # owned by
        'P112',  # founded by
        'P170',  # creator
        'P287',  # designed by
        'P84',   # architect
        'P176',  # manufacturer
        'P1056', # product or material produced
        'P180',  # depicts
        'P921',  # main subject
        'P2047', # duration
        'P2048', # height
        'P2049', # width
        'P2046', # area
        'P2043', # length
        'P2067', # mass
        'P2386', # diameter
        'P2044', # elevation above sea level
        'P1082', # population
        'P2046', # area
        'P2048', # height
        'P2067', # mass
        'P1120', # number of deaths
        'P1339', # number of injured
        'P1590', # unemployment rate
        'P2131', # nominal GDP
        'P2132', # nominal GDP per capita
        'P2133', # nominal GDP in purchasing power parity
        'P2134', # nominal GDP per capita in purchasing power parity
        'P2135', # GDP growth rate
        'P2136', # inflation rate
        'P2137', # industrial production growth rate
        'P2138', # labor force
        'P2139', # labor force participation rate
        'P2140', # Human Development Index
        'P2141', # Gini coefficient
        'P2142', # literacy rate
        'P2143', # life expectancy
        'P2144', # maternal mortality rate
        'P2145', # infant mortality rate
        'P2146', # birth rate
        'P2147', # death rate
        'P2148', # net migration rate
        'P2149', # urbanization rate
        'P2150', # total fertility rate
        'P2151', # Internet users
        'P2152', # telephone users
        'P2153', # mobile phone users
        'P2154', # electricity production
        'P2155', # electricity consumption
        'P2156', # oil production
        'P2157', # oil consumption
        'P2158', # natural gas production
        'P2159', # natural gas consumption
        'P2160', # carbon dioxide emissions
        'P2161', # military expenditure
        'P2162', # defense budget
        'P2163', # armed forces size
        'P2164', # number of households
        'P2165', # household size
        'P2166', # housing units
        'P2167', # occupied housing units
        'P2168', # vacant housing units
        'P2169', # home ownership rate
        'P2170', # median household income
        'P2171', # median family income
        'P2172', # per capita income
        'P2173', # poverty rate
        'P2174', # unemployment rate
        'P2175', # crime rate
        'P2176', # murder rate
        'P2177', # suicide rate
        'P2178', # traffic accident rate
        'P2179', # fire incident rate
        'P2180'  # natural disaster rate
    ]
    
    print("Preloading common properties...")
    fetch_properties_batch(common_properties)

def fetch_properties_batch(property_ids: List[str], batch_size: int = 50):# retrive multiples properties in one request
    global _property_labels_cache
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
   
    uncached_properties = [pid for pid in property_ids if pid not in _property_labels_cache]
    
    if not uncached_properties:
        return
    
    for i in range(0, len(uncached_properties), batch_size):
        batch = uncached_properties[i:i+batch_size]
        batch_ids = '|'.join(batch)
        
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbgetentities',
                'ids': batch_ids,
                'props': 'labels',
                'languages': 'en',
                'format': 'json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'entities' in data:
                with _cache_lock:
                    for property_id in batch:
                        if property_id in data['entities']:
                            entity_data = data['entities'][property_id]
                            if 'labels' in entity_data and 'en' in entity_data['labels']:
                                _property_labels_cache[property_id] = entity_data['labels']['en']['value']
                            else:
                                _property_labels_cache[property_id] = property_id
                        else:
                            _property_labels_cache[property_id] = property_id
            
            print(f"Fetched batch {i//batch_size + 1}: {len(batch)} properties")
            time.sleep(0.1)  
            
        except Exception as e:
            print(f"Error fetching batch {batch_ids}: {e}")
            with _cache_lock:
                for property_id in batch:
                    if property_id not in _property_labels_cache:
                        _property_labels_cache[property_id] = property_id

def get_property_label(property_id: str) -> str: #gets the label of a property from the cache
    with _cache_lock:
        return _property_labels_cache.get(property_id, property_id)

def collect_all_properties(g: Graph) -> Set[str]:# collect all properties wdt: used in the graph for entities with isWikidataNeighborOf
    properties = set()
    
    ex = Namespace("http://example.org/")
    entities_with_neighbor_relation = set()
    for subj, pred, obj in g.triples((None, ex.isWikidataNeighborOf, None)):
        entities_with_neighbor_relation.add(subj)
    
    for entity in entities_with_neighbor_relation:
        for subj, pred, obj in g.triples((entity, None, None)):
            pred_str = str(pred)
            if pred_str.startswith("http://www.wikidata.org/prop/direct/"):
                property_id = pred_str.split('/')[-1]
                properties.add(property_id)
    
    return properties


import time
import requests
from rdflib import Graph, Namespace
from typing import List, Dict, Set, Tuple
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

_property_labels_cache = {}
_cache_lock = threading.Lock()
_cache_file = 'property_cache.pkl'

def load_cache():
    global _property_labels_cache
    try:
        if os.path.exists(_cache_file):
            with open(_cache_file, 'rb') as f:
                _property_labels_cache = pickle.load(f)
            print(f"Cache loaded: {len(_property_labels_cache)} properties")
        else:
            _property_labels_cache = {}
    except Exception as e:
        print(f"Error loading cache: {e}")
        _property_labels_cache = {}

def save_cache():
    try:
        with open(_cache_file, 'wb') as f:
            pickle.dump(_property_labels_cache, f)
        print(f"Cache saved: {len(_property_labels_cache)} properties")
    except Exception as e:
        print(f"Error saving cache: {e}")

def fetch_properties_batch(property_ids: List[str], batch_size: int = 50):
    global _property_labels_cache
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Filtrer les propriétés non cachées
    uncached_properties = [pid for pid in property_ids if pid not in _property_labels_cache]
    
    if not uncached_properties:
        return
    
    # Traiter par batch avec multithreading
    def fetch_batch(batch):
        batch_ids = '|'.join(batch)
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbgetentities',
                'ids': batch_ids,
                'props': 'labels',
                'languages': 'en',
                'format': 'json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            batch_results = {}
            if 'entities' in data:
                for property_id in batch:
                    if property_id in data['entities']:
                        entity_data = data['entities'][property_id]
                        if 'labels' in entity_data and 'en' in entity_data['labels']:
                            batch_results[property_id] = entity_data['labels']['en']['value']
                        else:
                            batch_results[property_id] = property_id
                    else:
                        batch_results[property_id] = property_id
            
            return batch_results
            
        except Exception as e:
            print(f"Error fetching batch {batch_ids}: {e}")
            return {pid: pid for pid in batch}
    
    # Traiter les batches en parallèle
    batches = [uncached_properties[i:i+batch_size] for i in range(0, len(uncached_properties), batch_size)]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_batch = {executor.submit(fetch_batch, batch): batch for batch in batches}
        
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            with _cache_lock:
                _property_labels_cache.update(batch_results)
            print(f"Fetched batch: {len(batch_results)} properties")

# def get_property_label(property_id: str) -> str:
#     """Récupère le label d'une propriété depuis le cache"""
#     return _property_labels_cache.get(property_id, property_id)

def collect_all_data(g: Graph) -> Tuple[Set[str], Dict, List[Tuple]]:
    """Collecte toutes les données nécessaires en une seule passe"""
    properties = set()
    entity_labels = {}
    relations_data = []
    
    # Définir les namespaces
    ex = Namespace("http://example.org/")
    skos = Namespace("http://www.w3.org/2004/02/skos/core#")
    
    # Collecter les labels des entités
    print("Collecting entity labels...")
    start_time = time.time()
    for s, p, o in g.triples((None, skos.prefLabel, None)):
        entity_labels[s] = str(o)
    print(f"Found {len(entity_labels)} entity labels in {time.time() - start_time:.2f}s")
    
    # Trouver les entités avec la relation isWikidataNeighborOf
    print("Finding entities with neighbor relation...")
    start_time = time.time()
    entities_with_neighbor_relation = set()
    for subj, pred, obj in g.triples((None, ex.isWikidataNeighborOf, None)):
        entities_with_neighbor_relation.add(subj)
    print(f"Found {len(entities_with_neighbor_relation)} entities with neighbor relation in {time.time() - start_time:.2f}s")
    
    # Collecter toutes les relations wdt: en une seule passe
    print("Collecting all wdt relations...")
    start_time = time.time()
    processed_entities = 0
    
    for entity in entities_with_neighbor_relation:
        processed_entities += 1
        if processed_entities % 1000 == 0:
            elapsed = time.time() - start_time
            rate = processed_entities / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed_entities}/{len(entities_with_neighbor_relation)} entities ({rate:.0f} entities/s)")
        
        if entity not in entity_labels:
            continue
            
        for subj, pred, obj in g.triples((entity, None, None)):
            pred_str = str(pred)
            if pred_str.startswith("http://www.wikidata.org/prop/direct/"):
                property_id = pred_str.split('/')[-1]
                properties.add(property_id)
                
                # Stocker les données de la relation
                relations_data.append((entity, property_id, obj))
    
    collection_time = time.time() - start_time
    print(f"Found {len(relations_data)} wdt relations with {len(properties)} unique properties in {collection_time:.2f}s")
    print(f"Average rate: {len(relations_data) / collection_time:.0f} relations/s collected")
    
    return properties, entity_labels, relations_data

def process_relations_batch(relations_batch: List[Tuple], entity_labels: Dict, g: Graph) -> List[Dict[str, str]]:
    batch_start = time.time()
    results = []
    skos = Namespace("http://www.w3.org/2004/02/skos/core#")
    api_calls = 0
    
    print(f"  Processing batch of {len(relations_batch)} relations...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for i, (entity, property_id, destination_uri) in enumerate(relations_batch):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - batch_start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"    Batch progress: {i}/{len(relations_batch)} ({rate:.0f} relations/s in this batch)")
        
        source_label = entity_labels.get(entity)
        if not source_label:
            continue
        
        # Get property label from cache
        property_label = get_property_label(property_id)
        
        # Get destination entity label
        destination_label = entity_labels.get(destination_uri)
        
        # if  not found in cache, search in the graph
        if not destination_label:
            for s, p, o in g.triples((destination_uri, skos.prefLabel, None)):
                destination_label = str(o)
                break
        
        # Fallback vers Wikidata API (rare)
        if not destination_label:
            api_calls += 1
            uri_str = str(destination_uri)
            if 'wikidata.org' in uri_str:
                entity_id = uri_str.split('/')[-1]
                if entity_id.startswith('Q'):
                    try:
                        url = "https://www.wikidata.org/w/api.php"
                        params = {
                            'action': 'wbgetentities',
                            'ids': entity_id,
                            'props': 'labels',
                            'languages': 'en',
                            'format': 'json'
                        }
                        
                        response = requests.get(url, params=params, headers=headers, timeout=5)
                        response.raise_for_status()
                        data = response.json()
                        
                        if 'entities' in data and entity_id in data['entities']:
                            entity_data = data['entities'][entity_id]
                            if 'labels' in entity_data and 'en' in entity_data['labels']:
                                destination_label = entity_data['labels']['en']['value']
                            else:
                                destination_label = entity_id
                        else:
                            destination_label = entity_id
                            
                    except Exception:
                        destination_label = entity_id
                else:
                    destination_label = uri_str
            else:
                destination_label = uri_str
        
        if destination_label:
            results.append({
                'source': source_label,
                'destination': destination_label,
                'verbalization': property_label
            })
    
    batch_time = time.time() - batch_start
    print(f"  Batch completed: {len(results)} results in {batch_time:.2f}s ({len(results)/batch_time:.0f} relations/s), {api_calls} API calls")
    
    return results

def verbalize_rdf_relations(graph_path: str, max_workers: int = 8, batch_size: int = 1000) -> List[Dict[str, str]]:
    print("Starting ultra-optimized RDF verbalization...")
    start_time = time.time()
    
    load_cache()
    
    print("Loading RDF graph...")
    g = Graph()
    g.parse(graph_path, format="turtle")
    
    all_properties, entity_labels, relations_data = collect_all_data(g)
    
    print("Pre-fetching all property labels...")
    fetch_properties_batch(list(all_properties))
    
    save_cache()
    
    print(f"Data collection completed in {time.time() - start_time:.2f}s")
    print(f"Processing {len(relations_data)} relations...")
    
    relation_batches = [relations_data[i:i+batch_size] for i in range(0, len(relations_data), batch_size)]
    
    results = []
    processed_count = 0
    
    print(f"Starting parallel processing with {max_workers} workers, {len(relation_batches)} batches...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_relations_batch, batch, entity_labels, g): i 
            for i, batch in enumerate(relation_batches)
        }
        
        print(f"Submitted {len(future_to_batch)} batch jobs to executor")
        
        for i, future in enumerate(as_completed(future_to_batch)):
            batch_results = future.result()
            results.extend(batch_results)
            processed_count += len(batch_results)
            
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            
           
            if i % 10 == 0 or processed_count % 5000 == 0:
                print(f"Batch {i+1}/{len(relation_batches)} completed - Processed {processed_count}/{len(relations_data)} relations ({rate:.0f} relations/s) - Elapsed: {elapsed:.1f}s")
                
          
            if i % 5 == 0:
                print(f"  -> Batch {i+1} done: +{len(batch_results)} relations")
    
    total_time = time.time() - start_time
    print(f"Completed! Generated {len(results)} verbalizations in {total_time:.2f}s")
    print(f"Average rate: {len(results) / total_time:.0f} relations/s")
    

    save_cache()
    
    return results

# to see cache size
def get_cache_stats():
    return f"Cache contains {len(_property_labels_cache)} properties"

def clear_cache():
    global _property_labels_cache
    _property_labels_cache.clear()
    if os.path.exists(_cache_file):
        os.remove(_cache_file)
    print("Cache cleared")