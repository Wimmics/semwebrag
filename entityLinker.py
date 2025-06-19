import spacy
import numpy as np
from spacy.tokens import Span
import re
from rdflib import Graph, Literal, URIRef, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD
import os
from spacy.kb import KnowledgeBase
from spacy.kb import InMemoryLookupKB
from rdflib.namespace import SKOS


REL = Namespace("http://relations.example.org/")
WD = Namespace("http://www.wikidata.org/wiki/")

def divide_text(text, max_size=999999):# to divide the text into parts of max_size characters
    res = []
    if len(text) <= max_size:
        res.append(text)
        return res
    else:
        print("DIVISION")
        res.append(text[:max_size])
        text = text[max_size:]
        res += divide_text(text, max_size)
        return res
    


def chunk_text(text, nlp):
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


def add_entity_linked_to_graph(graph_path,graph_destination, text):
    REL = Namespace("http://relations.example.org/")
    WD = Namespace("http://www.wikidata.org/wiki/")

    nlp = spacy.load("en_core_web_md")

    print(nlp.vocab)

    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=384)

    nlp.add_pipe("entityLinker", last=True)

    graph = Graph()
    graph.parse(graph_path, format='turtle')

    text = re.sub(r'\[.*\]', '', text)


    textparts = divide_text(text)
    print("divisé")
    chunks = []
    for part in textparts:
        chunks += chunk_text(part, nlp)
    print("chunké")
    print("chunks : ", chunks)
    chunkID = 0
    for chunk in chunks:
        doc = nlp(chunk)
        print("\n chunk actuel : ", chunk)
        print("\n doc actuel : ", doc.ents)
        linked_entities = doc._.linkedEntities

        for entity in linked_entities:
            entity_id = entity.get_id()
            entity_label = entity.get_label()
            entity_description = entity.get_description()
            entity_url = entity.get_url()

            print("entity id : ", entity_id)
            print("entity label : ", entity_label)
            print("entity description : ", entity_description)
            print("entity url : ", entity_url)

            entity_uri = URIRef(entity_url)

            if not any(s == entity_uri for s, p, o in graph):#ajouter l'entité dans la graphe si uelle y est pas déjà
                # entityForURIRef = [re.sub(r'[^a-zA-Z0-9]', '_', ent.text.lower()) for ent in entities]
                if entity_label == None:
                    entity_label = ""
                else:
                    entity_name = entity_label.replace(" ", "_")

                uri = URIRef(f"http://example.org/entity/{entity_name}")
                print(f"Ajout de l'entité {entity_name} ({entity_id}) au graphe.")
                graph.add((uri, RDF.type, URIRef(entity_uri)))  # Exemple de type, peut être ajusté selon le domaine
                graph.add((uri, SKOS.prefLabel, Literal(entity_label)))
                graph.add((uri, RDFS.comment, Literal(entity_description)))
                #graph.add((uri, URIRef("http://www.w3.org/2000/01/rdf-schema#seeAlso"), Literal(entity_url)))
                #graph.add((uri, REL.mentionedIn, URIRef(f"http://example.org/chunk_{chunkID}")))
                
                
            else:
                print(f"L'entité {entity_label} ({entity_id}) est déjà présente dans le graphe.")
                #si l'entité n'a pas de RDF.comment, on l'ajoute
                if not any((s, p, o) for s, p, o in graph.triples((entity_uri, RDFS.comment, None))):
                    graph.add((entity_uri, RDFS.comment, Literal(entity_description)))
        chunkID += 1

    graph.serialize(destination= graph_destination, format='turtle')




def link_wikiData_entities_to_chunks(graph_path, graph_destination):
    graph = Graph()
    graph.parse(graph_path, format='turtle')

    
    for entity, _, entity_type in graph.triples((None, RDF.type, None)):
        if "wikidata" in str(entity_type):
            entity_pref_label = None
            for _, _, label in graph.triples((entity, SKOS.prefLabel, None)):
                entity_pref_label = str(label).strip().lower()
                break

            if not entity_pref_label:
                continue  # skip if no prefLabel

            print(f"Traitement de l'entité : {entity_pref_label}")

            # Parcourir tous les chunks
            for chunk, _, chunk_type in graph.triples((None, RDF.type, URIRef("http://example.org/Chunk"))):
                # Récupérer le prefLabel du chunk
                chunk_pref_label = None
                for _, _, label in graph.triples((chunk, SKOS.prefLabel, None)):
                    chunk_pref_label = str(label).strip().lower()
                    break

                if not chunk_pref_label:
                    continue  # skip if no  prefLabel for the chunk

                # chinking if prefLabel of entity is in chunk prefLabel
                if entity_pref_label in chunk_pref_label:
                    print(f" {entity} rel:mentionedIn {chunk}")
                    graph.add((entity, REL.mentionedIn, chunk))

    graph.serialize(destination=graph_destination, format='turtle')