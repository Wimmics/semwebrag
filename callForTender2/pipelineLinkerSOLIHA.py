from rdflib import Graph, RDF, RDFS, Literal, URIRef, Namespace
import time
from langchain.embeddings import HuggingFaceEmbeddings
from rdflib.namespace import SKOS
from rdflib.namespace import OWL 
import spacy
import numpy as np
import re
import requests
from spacy.tokens import Span
from spacy.language import Language
import wikidatautils 
import entityLinker 
import json

from DAO import FaissDAO

t0 = time.time()
local_model_path = ".embeddings/models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=local_model_path)
temps = time.time()-t0
print("embeddings loaded in : ", time.time()-t0, "s")

nlp = spacy.load("fr_core_news_md")
    
text = ""

with open("callForTender2/SOLIHAdata.txt", "r", encoding="utf-8") as file:
    text = file.read()

REL = Namespace("http://relations.example.org/")
WD = Namespace("http://www.wikidata.org/wiki/")

# prendre les entités et labels de l'ontologie
# def extract_entities(ontology_file):
#     g = Graph()
#     g.parse(ontology_file, format="turtle")
   
#     entities_list = []
#     labels_list = []
    
#     # prendre les entités qui ont un skos:prefLabel
#     # for s in g.subjects(RDF.type, None):
#     #     pref_label = g.value(s, SKOS.prefLabel)
#     #     if pref_label and isinstance(pref_label, Literal):
#     #         entities_list.append(s)
#     #         labels_list.append(pref_label.value)

#     # prendre les entités qui ont un rdfs:label
#     for s in g.subjects(RDF.type, None):
#         pref_label = g.value(s, RDFS.label)
#         if pref_label and isinstance(pref_label, Literal):
#             entities_list.append(s)
#             labels_list.append(pref_label.value)

#     return entities_list, labels_list


def extract_entities(ontology_file):
    g = Graph()
    g.parse(ontology_file, format="turtle")
   
    entities_list = []
    labels_list = []
    
    print(f"extraction des entités depuis : {ontology_file}")
    entity_count = 0
    
    # prendre les entités qui ont un rdfs:label
    for s in g.subjects(RDF.type, None):
        entity_count += 1
        print(f"\nEntité #{entity_count}: {s}")
        
        # Récupérer tous les labels pour cette entité
        all_labels = list(g.objects(s, RDFS.label))
        print(f"  Tous les labels trouvés: {len(all_labels)}")
        # Chercher spécifiquement le label en français
        french_label = None
        for label in all_labels:
            if isinstance(label, Literal):
                print(f"    Label: '{label}' (langue: {label.language})")
                if label.language == 'fr':
                    french_label = label
                    print(f"    → Label français trouvé: '{french_label}'")
                    break
        if french_label:
            entities_list.append(s)
            labels_list.append(french_label.value)
            print(f"Entité ajoutée : '{french_label.value}'")
        else:
            print(f"pas de label francais trouvé ")
    
    print(f"\n=== RÉSULTATS ===")
    print(f"Nombre total d'entités examinées: {entity_count}")
    print(f"Nombre d'entités avec label français: {len(entities_list)}")
    print(f"Labels français extraits:")
    for i, label in enumerate(labels_list, 1):
        print(f"  {i}. {label}")
    
    return entities_list, labels_list




# créer une liste d'embeddings pour chaque label d'entité
def get_ontology_embeddings(embeddings, entityLabels):
    ontologyEmbeddings = []
    print("Extraction des embeddings pour l'ontologie ...")
    for name in entityLabels:
        embedding = embeddings.embed_query(name)
        ontologyEmbeddings.append(embedding)

    #le nombre d'embeddings doit correspondr au nombre de labels
    if len(ontologyEmbeddings) != len(entityLabels):
        print("erreur nombre d'embeddings/labels")
    else:
        print("===================== embeddings ontologie ok. ======================")
  

    return ontologyEmbeddings


# print("NER ...")
# doc = nlp(text)


def entityRetriever(embedding, ontologyEmbeddings,entities): # à partir d'un embedding, trouver l'entité de l'ontologie la plus proche
    best_similarity = -1
    best_entity = None
    for i, e in enumerate(ontologyEmbeddings):
        similarity = np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e))
        if similarity > best_similarity:
            best_similarity = similarity
            best_entity = entities[i]
    return best_entity, best_similarity

# def chunk_text(text):
#     # 1 chunk = une phrase
#     doc = nlp(text)
#     chunks = [sent.text for sent in doc.sents]
#     return chunks

def chunk_text(text):
    # 1 chunk = une ligne
    lines = text.split('\n')
    chunks = [line.strip() for line in lines if line.strip()]  
    print (f"Nombre de chunks extraits: {len(chunks)}")
    return chunks


def extract_key_phrases(doc, nlp):
    # ner
    entities = list(doc.ents)


    print("phrases  AAAAAAAAAAAAAA:   ",doc.ents)
    
    # prendre les chunk avec + de 1 mot
    noun_chunks = [chunk for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    
    # ajouter les chunks aux entités
    all_entities = entities + noun_chunks
    
    # enlever les doublons
    unique_entities = []
    seen_texts = set()
    
    for ent in all_entities:
        normalized_text = ent.text.strip().lower()
        if normalized_text not in seen_texts and len(normalized_text) > 2:
            seen_texts.add(normalized_text)
            unique_entities.append(ent)
    
    return unique_entities


def extract_entity_from_query(text) :
    unique_entities = []
    
    doc = nlp(text)
    entities = list(doc.ents)
    print("Entités extraites de la requête:")
    for ent in entities:
        print(f"- {ent.text} ({ent.label_})")
        normalized_text = ent.text.strip().lower()
        if normalized_text not in unique_entities and len(normalized_text) > 3:
            unique_entities.append(normalized_text)
    print(f"Nombre total d'entités uniques extraites de la requête: {len(unique_entities)}")
    return unique_entities




# def extract_key_phrases(chunks, nlp):
#     """
#     Extrait les phrases clés de tous les chunks de texte
#     chunks: liste de chaînes de caractères (retour de chunk_text)
#     nlp: modèle spaCy
#     """
#     all_entities = []
    
#     # Traiter chaque chunk individuellement
#     for chunk in chunks:
#         if not chunk.strip():  # ignorer les chunks vides
#             continue
            
#         # Analyser le chunk avec spaCy
#         doc = nlp(chunk)
        
#         # Extraire les entités nommées
#         entities = list(doc.ents)
#         print(f"Entités trouvées dans '{chunk[:50]}...': {[ent.text for ent in entities]}")
        
#         # Extraire les chunks nominaux avec plus d'1 mot
#         noun_chunks = [chunk_obj for chunk_obj in doc.noun_chunks if len(chunk_obj.text.split()) > 1]
        
#         # Ajouter à la liste globale
#         all_entities.extend(entities + noun_chunks)
    
#     # Enlever les doublons
#     unique_entities = []
#     seen_texts = set()
    
#     for ent in all_entities:
#         normalized_text = ent.text.strip().lower()
#         if normalized_text not in seen_texts and len(normalized_text) > 3:
#             seen_texts.add(normalized_text)
#             unique_entities.append(ent)
    
#     print(f"Nombre total d'entités uniques extraites: {len(unique_entities)}")
#     return unique_entities


def find_entity_in_chunks(entity_text, chunks):
    entity_text_lower = entity_text.lower()
    chunk_indices = []
    
    for i, chunk in enumerate(chunks):
        if entity_text_lower in chunk.lower():
            chunk_indices.append(i)
    
    return chunk_indices

def build_knowledge_graph_aligned_with_ontology(text,ontology_path, nlp, rdf_path, embeddings):
    wikidataLabelList = []
    doc = nlp(text)
    chunks = chunk_text(text)
    time.sleep(5)
    DAO = FaissDAO(384)

    print("Extraction des entités et labels de l'ontologie ...")
    ontologyEntities, entityLabels = extract_entities(ontology_path)

    entityNames = [str(e).split("/")[-1] for e in ontologyEntities]
    print("entityNames (premiers 5):", entityNames[3001:3006] if len(entityNames) >= 5 else entityNames)
    print("entityNames size:", len(entityNames))
    print("entityLabels (premiers 5):", list(entityLabels)[3001:3006] if len(entityLabels) >= 5 else list(entityLabels))
    print("entityLabels size:", len(entityLabels))

    # vérifier que les deux listes ont la même taille
    if len(ontologyEntities) != len(entityLabels):
        print("erreur nombre d'entités/nombre de labels")
    else:
        print("le nombre d'enittés/labels est le meme")

    ontologyEmbeddings = get_ontology_embeddings(embeddings, entityLabels)
    
    print(f"Nombre d'entités: {len(ontologyEntities)}")
    print(f"Nombre de labels: {len(entityLabels)}")
    print(f"Nombre d'embeddings: {len(ontologyEmbeddings)}")

        
    # NER
    entities = extract_key_phrases(doc, nlp)
        
    print("Entités du texte:")
    for ent in entities:
        print(f"- {ent.text} ({ent.label_ if hasattr(ent, 'label_') else 'PHRASE'})")

    # remplacer les espaces dans les noms des entités par des _, retirer les caractères spéciaux et mettre en minuscule
    entityForURIRef = [re.sub(r'[^a-zA-Z0-9]', '_', ent.text.lower()) for ent in entities]
    
    print("Construction du graphe ...")
    # Créer un graphe RDF
    g = Graph()
    
    # Définir les namespaces
    ATC = Namespace("http://purl.bioontology.org/ontology/ATC/")
    g.bind("rel", REL)
    g.bind("atc", ATC)
    g.bind("wd", WD)
    
    # mettre  les chunks dans le graphe
    chunk_uris = []
    for i, chunk in enumerate(chunks):
        chunk_uri = URIRef(f"http://example.org/chunk_{i}")
        g.add((chunk_uri, RDF.type, URIRef("http://example.org/Chunk")))
        g.add((chunk_uri, SKOS.prefLabel, Literal(chunk)))
        chunk_uris.append(chunk_uri)
    
    # Ajouter les entités du texte et les relier aux chunks
    for i, ent in enumerate(entities):
        print("inesertion des entités et leur embedding dans le DAO")
        DAO.insert(ent.text, embeddings.embed_query(ent.text))
        # Créer URI pour l'entité extraite du texte
        entity_uri = URIRef(f"http://example.org/entity/{entityForURIRef[i]}")
        
        #récupérer l'entité correspondante
        ontology_entity, similarity = entityRetriever(embeddings.embed_query(ent.text), ontologyEmbeddings,ontologyEntities)
        
        # Ajouter l'entité au graphe
        g.add((entity_uri, SKOS.prefLabel, Literal(ent.text)))
        
        #si la similarité avec ATC est suffisante, lier à ATC
        if similarity >= 0.5:
            g.add((entity_uri, RDF.type, URIRef(ontology_entity)))
            g.add((entity_uri, REL.alignmentScore, Literal(similarity)))
            g.add((entity_uri, REL.alignedWith, URIRef(ontology_entity)))
            print(f"Entity '{ent.text}' aligned with the ontology : {ontology_entity} (score: {similarity:.4f})==============================================================================")
           
        else:
            # wikidata_entity = wikidatautils.get_uri_wikidata(ent.text) 
            # g.add((entity_uri, RDF.type, URIRef(wikidata_entity)))
            # wikidatalabel = wikidatautils.get_description_from_entity(wikidata_entity)
            # print("label : ", wikidatalabel)
            # g.add((entity_uri, SKOS.prefLabel, Literal(wikidatalabel)))
            # wikidataLabelList.append(wikidatalabel)
            print(f"Entity '{ent.text}' not aligned with with the current ontology.")
        
        # Trouver les chunks qui contiennent cette entité et les relier
        chunk_indices = find_entity_in_chunks(ent.text, chunks)
        for idx in chunk_indices:
            g.add((entity_uri, REL.mentionedIn, chunk_uris[idx]))
            g.add((chunk_uris[idx], REL.mentions, entity_uri))

   
    g.serialize(destination=rdf_path, format="turtle")
    entityLinker.add_entity_linked_to_graph(rdf_path, "callForTender2/outputLinker.ttl", text)
    entityLinker.link_wikiData_entities_to_chunks("callForTender2/outputLinker.ttl", "callForTender2/outputLinkerLinked_tmp.ttl")

    #ajouter les voisins direct sur wikidata des entités et les lier à l'entité de base
    _, neighborList = wikidatautils.add_wikidata_neighbors_to_graph("callForTender2/outputLinkerLinked_tmp.ttl", output_path="callForTender2/outputLinkerLinked.ttl" )  
    # owlThingList = remove_useless_owl_things("finance/outputLinkerLinked.ttl", "finance/outputLinkerLinked.ttl")

    #ajouter les labels de owl:Thing restants dans via le dao
    for label in neighborList:
        DAO.insert(label, embeddings.embed_query(label))


    #g.serialize(destination=rdf_path, format="turtle")
    print(f"Graphe sauvegardé, {len(g)} triplets.")
    DAO.save_index("callForTender2/embeddings.index")
    convert_wikidata_with_regex("callForTender2/outputLinkerLinked.ttl", "callForTender2/outputLinkerLinked.ttl")
    return g


import re

def convert_wikidata_with_regex(input_file, output_file):

  
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    

    pattern = r'wd:Q(\d+)(?![a-zA-Z])'
    

    transformed = re.sub(pattern, r'<https://www.wikidata.org/wiki/Q\1>', content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transformed)
    
    print(f"Conversion terminée : {output_file} créé avec succès.")



def remove_useless_owl_things(input_file, output_file):

    g = Graph()
    g.parse(input_file, format="ttl")
   
    owl_things = list(g.subjects(RDF.type, OWL.Thing))
   
    to_remove = []
    for thing in owl_things:
        prefLabels = list(g.objects(thing, SKOS.prefLabel))
        if len(prefLabels) == 0:
            to_remove.append(thing)
            
    print(f"Total owl:Thing: {len(owl_things)}")
    print(f"owl:Thing à supprimer (sans prefLabel): {len(to_remove)}")
   
    for thing in to_remove:
        g.remove((thing, None, None))
        g.remove((None, None, thing))
   
    remaining = list(g.subjects(RDF.type, OWL.Thing))
    print(f"owl:Thing restants: {len(remaining)}")
   
    g.serialize(destination=output_file, format="ttl")
    
    remaining_preflabels = []
    for thing in remaining:
        for label in g.objects(thing, SKOS.prefLabel):
            remaining_preflabels.append((str(thing), str(label)))
    
    for i, (thing, label) in enumerate(remaining_preflabels):
        print(f"Remaining owl:Thing {i+1}: {thing} with label: {label}")
    
    return remaining_preflabels



def extract_labels_from_graph(graph):
    labelList = []
    for s, p, o in graph.triples((None, SKOS.prefLabel, None)):
        if isinstance(o, Literal):
            labelList.append(o.value)

    return labelList

def get_embeddings_from_labels(labelList):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddingsList = []
    for name in labelList:
        embedding = embeddings.embed_query(name)
        embeddingsList.append(embedding)
    
    return embeddingsList

def retrieve_corresponding_label(entity, embeddingsList, labelList):
    embedding = embeddings.embed_query(entity.text)
    best_similarity = -1
    best_entity = None
    for i, e in enumerate(embeddingsList):
        similarity = np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e))
        if similarity > best_similarity:
            best_similarity = similarity
            best_entity = labelList[i]

    return best_entity

def process_query(query_text, rdf_graph_path, embeddings_model=embeddings, output_file="callForTender2/query_enrichie.txt", neighborChunks=0): #traite la query, trouve les entités pertinente dans le graphe et enrichit la query avec les chunks liés
    tload = time.time()
    DAO = FaissDAO(384)
    DAO.load_index("callForTender2/embeddings.index")
    g = Graph()
    g.parse(rdf_graph_path, format="turtle")
    g.bind("rel", REL)

    tloaded = time.time()-tload
    
    #extraire les entités de la query
    textract = time.time()
    doc = nlp(query_text)
    query_entities = extract_key_phrases(doc, nlp)
    
    enriched_results = ["question :","\n\n", query_text, "\n\n"]
    enriched_results.append("context : ")
    enriched_results.append("\n\n")

    labelsList = extract_labels_from_graph(g)
    print("nombre d'entités dans le graphe : ", len(labelsList))
    #print("labelsList : ", labelsList)
    #_
    #embeddingsList = get_embeddings_from_labels(labelsList)
    #_
    #print ("embeddingsList : ", embeddingsList)
    chunks_already_mentioned = set()
    chunkList = []
    # l = []
    
    textracted = time.time()-textract

    print("dernière partie ...")
    t = time.time()
    wikidatautils.initialize_json_file("callForTender2/logs.json")
    #pour chaque entité dans la requete
    for ent in query_entities:
        print(f"ent : {ent.text}")
        # print("ent : ", ent.text)

        #correspondingEnt = retrieve_corresponding_label(ent, embeddingsList, labelsList)
        correspondingEnt, distance = DAO.search(embeddings.embed_query(ent.text), k=1)

        print("correspondingEnt de : ",ent.text," ", correspondingEnt)
        
        l, chunkList = wikidatautils.retrieve_mentioned_chunks(rdf_graph_path, correspondingEnt[0], chunkList, neighborChunks)
        print("CHUNK COUNT : ", len(chunkList))
        entity_data = {
            'entity': ent.text,
            'correspondingEnt': correspondingEnt,
            'chunkCount': l  
        }
        wikidatautils.add_to_json_file("callForTender2/logs.json", entity_data)

    for chunk in chunkList:
        enriched_results.append(chunk)
    print("chunks_already_mentioned : ", chunks_already_mentioned)

    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(enriched_results))

    print("derniere partie finie en : ", time.time()-t, "s")


    print("temps chargment : ", tload, "s")
    print("temps extraction : ", textract, "s")
    
    print(f"Requête enrichie sauvegardée dans {output_file}")
    return "\n".join(enriched_results)

# à commenter pour pas reconstruire le graphe
# build_knowledge_graph_aligned_with_ontology(text, "callForTender2/public-contracts.ttl", nlp, "callForTender2/knowledge_graphNoWiki.ttl", embeddings)


 
# process_query("Que peux tu me dire sur l'ARTICLE 8 ? Quelles en sont les clauses ? Connais tu la matmut ?  ","callForTender/outputLinkerLinked.ttl", neighborChunks=4)


print("temps d'importation : ", temps, "s")

print("temps total : ", time.time()-t0, "s")