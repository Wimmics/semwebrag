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

nlp = spacy.load("en_core_web_md")
    
text = ""

with open("financeClassic/financialText.txt", "r", encoding="utf-8") as file:
    text = file.read()



def chunk_text(text):
    # 1 chunk = une phrase
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    return chunks

def extract_key_phrases(doc, nlp):
    # ner
    entities = list(doc.ents)
    
    # prendre les chunk avec + de 1 mot
    noun_chunks = [chunk for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    
    # ajouter les chunks aux entités
    all_entities = entities + noun_chunks
    
    # enlever les doublons
    unique_entities = []
    seen_texts = set()
    
    for ent in all_entities:
        normalized_text = ent.text.strip().lower()
        if normalized_text not in seen_texts and len(normalized_text) > 3:
            seen_texts.add(normalized_text)
            unique_entities.append(ent)
    
    return unique_entities


def build_knowledge_graph_aligned_with_ontology(text,ontology_path, nlp, rdf_path, embeddings):
    doc = nlp(text)
    chunks = chunk_text(text)
    DAO = FaissDAO(384)

    #ajouter les chunks et les embeddings dans le dao
    for chunk in chunks:
        chunk = chunk.strip()
        print(f"Chunk : {chunk[:50]}...")  # Afficher les 50 premiers caractères du chunk
        if len(chunk) > 0:
            
            DAO.insert(chunk, embeddings.embed_query(chunk))


    #g.serialize(destination=rdf_path, format="turtle")
    print(f"Graphe sauvegardé, triplets.")
    DAO.save_index("financeClassic/embeddings.index")
    


import re

def process_query(query_text, rdf_graph_path, embeddings_model=embeddings, output_file="financeClassic/query_enrichie.txt", neighborChunks=0): #traite la query, trouve les entités pertinente dans le graphe et enrichit la query avec les chunks liés
    doc = nlp(text)
    tload = time.time()
    DAO = FaissDAO(384)
    DAO.load_index("financeClassic/embeddings.index")
    chunks_already_mentioned = []
    tloaded = time.time()-tload
    
    #extraire les entités de la query
    textract = time.time()


    enriched_results = ["question :","\n\n", query_text, "\n\n"]
    enriched_results.append("context : ")
    enriched_results.append("\n\n")

 
    
    textracted = time.time()-textract

    print("dernière partie ...")
    t = time.time()
    wikidatautils.initialize_json_file("financeClassic/logs.json")
    #pour chaque entité dans la requete
    embedded_query = embeddings_model.embed_query(query_text)
    #chergcher et afficher les nChunks les plus proches
    correspondingEnt, distance = DAO.search(embedded_query, k=1)

    if(neighborChunks > 0):
        chunks = chunk_text(text)
        #chercher les nchunks en haut et en bas par rapport au chunk trouvé dans la liste
        chunk_index = chunks.index(correspondingEnt[0]) if correspondingEnt[0] in chunks else -1
        # if chunk_index != -1:
        #     continue
        for i in range(1, neighborChunks):
            if chunk_index - i >= 0:
                correspondingEnt.append(chunks[chunk_index - i])
                distance.append(0)
            if chunk_index + i < len(chunks):
                correspondingEnt.append(chunks[chunk_index + i])

    for i, ent in enumerate(correspondingEnt):
        print(f"Entité {i+1} : {ent} (distance : ")#{distance[i]})")
        if ent not in chunks_already_mentioned:
            enriched_results.append(f"{ent}\n")
            chunks_already_mentioned.append(ent)

    entity_data = {
        'entity': query_text,
        'correspondingEnt': correspondingEnt,
        'chunkCount': correspondingEnt  
        }
    wikidatautils.add_to_json_file("financeClassic/logs.json", entity_data)


    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(enriched_results))

    print("derniere partie finie en : ", time.time()-t, "s")


    print("temps chargment : ", tload, "s")
    print("temps extraction : ", textract, "s")
    
    print(f"Requête enrichie sauvegardée dans {output_file}")
    return "\n".join(enriched_results)

# à commenter pour pas reconstruire le graphe
# build_knowledge_graph_aligned_with_ontology(text, "financeClassic/dev.fibo-quickstart.ttl", nlp, "financeClassic/knowledge_graphNoWiki.ttl", embeddings)
# remove_useless_owl_things("finance/outputLinkerLinked.ttl", "finance/outputLinkerLinked.ttl")

# process_query("What are the main domains of NVIDIA","finance/outputLinkerLinked.ttl", embeddings, neighborChunks=5)


print("temps d'importation : ", temps, "s")

print("temps total : ", time.time()-t0, "s")