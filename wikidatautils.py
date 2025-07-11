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
            time.sleep(1)  #pause pour éviter erreur 429
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

    import rdflib
    from rdflib import URIRef, Literal, Graph
    from rdflib.namespace import RDF, RDFS, OWL, SKOS
    import time
    
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
    
    #For each entity seach neighbors with label and add them to the graph
    for row in results:
        entity_uri = row.entity
        wikidata_uri = row.type
        wikidata_id = str(wikidata_uri).split("/")[-1]
        
        print(f"entité: {entity_uri} (Wikidata: {wikidata_id})")
        
        # outgoing relations
        sparql_query_out = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT ?predicate ?object ?objectLabel
                WHERE {{
                SERVICE <https://query.wikidata.org/sparql> {{
                    wd:{wikidata_id} ?predicate ?object .
                    FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
                    ?object rdfs:label ?objectLabel .
                    FILTER(LANG(?objectLabel) = "fr" || LANG(?objectLabel) = "en")
                }}
                }}
                #LIMIT 1000
        """
        
        # incoming relations
        sparql_query_in = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        SELECT ?predicate ?subject ?subjectLabel
        WHERE {{
          SERVICE <https://query.wikidata.org/sparql> {{
            ?subject ?predicate wd:{wikidata_id} .
            FILTER(STRSTARTS(STR(?predicate), "http://www.wikidata.org/prop/"))
            ?subject rdfs:label ?subjectLabel .
            FILTER(LANG(?subjectLabel) = "fr" || LANG(?subjectLabel) = "en")
          }}
        }}
        LIMIT 1000
        """
        
        try:
            results_out = graph.query(sparql_query_out)
            print(f"RESULTS OUT: {results_out}, TAILLE : {len(results_out)}")
            results_in = graph.query(sparql_query_in)
            print(f"RESULTS IN: {results_in}, TAILLE : {len(results_in)}")
            
            total_results = len(results_out) + len(results_in)
            print(f"{total_results} relations trouvées")
            
            entity_count = 0
            
            for result in results_out:

               
                predicate_uri = str(result.predicate)
               
               
                object_value = str(result.object)
                
             

                clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
                clean_predicate = clean_predicate.replace("direct/", "")
                clean_predicate_uri = WDT[clean_predicate]
                
                if isinstance(result.object, URIRef):
                    obj = URIRef(object_value)
                     
                    graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, obj))
                    
                    graph.add((obj, RDF.type, OWL.Thing))
                    
                    #add isWikidataNeighborOf relation used when we want to get chunks linked to the entity
                    graph.add((obj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                    graph.add((obj, EX.relationDirection, Literal("outgoing")))
                    
                    #add label
                    label = str(result.objectLabel)
                    graph.add((obj, SKOS.prefLabel, Literal(label)))
                    if(label not in labelList):
                        labelList.append(label)
                    print(f"Ajouté le label '{label}' pour l'entité {obj}")
                    entity_count += 1
                else:
                    print("!!!===================ALERT LITTERAL VALUE =========================================")
                    print(Literal(object_value))
                    #if it is a litteral value, add it directly
                    graph.add((URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}"), clean_predicate_uri, Literal(object_value)))
            
            for result in results_in:
                subject_uri = str(result.subject)
                predicate_uri = str(result.predicate)
                subject_label = str(result.subjectLabel)
                if(label not in labelList):
                    labelList.append(subject_label)
                print(f"Ajouté le label '{subject_label}' pour l'entité {subject_uri}")
                
                clean_predicate = predicate_uri.replace("http://www.wikidata.org/prop/", "")
                clean_predicate = clean_predicate.replace("direct/", "")
                clean_predicate_uri = WDT[clean_predicate]
                
                subj = URIRef(subject_uri)
                graph.add((subj, clean_predicate_uri, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                
                graph.add((subj, RDF.type, OWL.Thing))
                
                graph.add((subj, EX.isWikidataNeighborOf, URIRef(f"http://www.wikidata.org/wiki/{wikidata_id}")))
                graph.add((subj, EX.relationDirection, Literal("incoming")))
                
                graph.add((subj, SKOS.prefLabel, Literal(subject_label)))
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

def filter_neigbor(graph_path, property_stats_path, output_path="filtered_graph.ttl", threshold=20):
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

    #prendre les labels des entités de resList
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
    # Extraire l'ID Qxxxx
    match = re.search(r'Q\d+', str(wikidata_uri))
    if not match:
        return {"label": None, "description": None, "aliases": []}

    qid = match.group(0)

    # SPARQL vers Wikidata
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

    # Coupe la description si trop longue
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
    
    # Find entity URI by label
    entity_uri = None
    for s, p, o in g.triples((None, SKOS.prefLabel, Literal(label))):
        entity_uri = s
        break
    if not entity_uri:
        for s, p, o in g.triples((None, RDFS.label, Literal(label))):
            entity_uri = s
            break
    
    if not entity_uri:
        return f"Entity '{label}' has no complementary information."
    
    # Get entity type
    entity_type = None
    for s, p, o in g.triples((entity_uri, RDF.type, None)):
        entity_type = o
        break
    
    if not entity_type:
        return f"Entity '{label}' has no complementary information."
    
    # Check if it's a Wikidata entity
    if str(entity_type).startswith("https://www.wikidata.org/wiki/Q"):
        wikidata_id = str(entity_type).split("/")[-1]
        
        # with P31
        sparql_query_with_type = f"""
        SELECT ?typeLabel ?description ?altLabel WHERE {{
          wd:{wikidata_id} wdt:P31 ?type .
          wd:{wikidata_id} schema:description ?description .
          OPTIONAL {{ wd:{wikidata_id} skos:altLabel ?altLabel . }}
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en" .
            ?type rdfs:label ?typeLabel .
          }}
          
          FILTER(LANG(?description) = "en")
          FILTER(LANG(?altLabel) = "en" || !BOUND(?altLabel))
        }}
        LIMIT 10
        """
        
        #Only description and aliases without type
        sparql_query_without_type = f"""
        SELECT ?description ?altLabel WHERE {{
          wd:{wikidata_id} schema:description ?description .
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
            
            # Try with type first
            response = requests.get(
                endpoint,
                params={'query': sparql_query_with_type, 'format': 'json'},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                
                if bindings:
                    result = bindings[0]
                    
                    type_label = result.get('typeLabel', {}).get('value', 'Unknown')
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
                    
                    return f"{label} is of type {type_label}, as description: {description}, is also known as: {alt_labels_str}"
                
                # of no P31 found, try without type
                else:
                    response = requests.get(
                        endpoint,
                        params={'query': sparql_query_without_type, 'format': 'json'},
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        bindings = data.get('results', {}).get('bindings', [])
                        
                        if bindings:
                            result = bindings[0]
                            
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
                            
                            return f"{label} (type unknown), as description: {description}, is also known as: {alt_labels_str}"
                    
        except Exception as e:
            pass
    
    # Check if it's an ontology entity
    for s, p, o in ontology.triples((entity_type, None, None)):
        # Found in ontology, get label
        for s2, p2, o2 in ontology.triples((entity_type, SKOS.prefLabel, None)):
            return f"{label} is of type {o2}"
        for s2, p2, o2 in ontology.triples((entity_type, RDFS.label, None)):
            return f"{label} is of type {o2}"
        # If no label found, use URI
        return f"{label} is of type {str(entity_type)}"
    
    # Default case
    return f"Entity '{label}' has no complementary information."



import requests
import time
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, SKOS
import difflib

def align_unlinked_entities_to_wikidata(ttl_file_path: str, output_path: str = None):
    #try to align with wikidata entities with rel:mentioned in not aligned with either ontology or wikidata

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
                    
                    # add triplet
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
        
        # Délai pour éviter de surcharger l'API
        time.sleep(0.1)
    
    # Sauvegarder le graphe modifié
    if output_path is None:
        base_name = ttl_file_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_aligned.ttl"
    
    g.serialize(destination=output_path, format='turtle')
    print(f"Alignement terminé: {aligned_count} entités alignées")
    print(f"Fichier sauvegardé: {output_path}")
    
    return output_path
