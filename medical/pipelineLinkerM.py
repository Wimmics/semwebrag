from rdflib import Graph, RDF, Literal, URIRef, Namespace
from langchain.embeddings import HuggingFaceEmbeddings
from rdflib.namespace import SKOS
import spacy
import numpy as np
import re
import requests
from spacy.tokens import Span
from spacy.language import Language
import wikidatautils 
import entityLinker 
from DAO import FaissDAO   
from DAO_relations import FaissDAO_relations
import time

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
local_model_path = ".embeddings/models/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=local_model_path)


nlp = spacy.load("en_core_web_md")
    
text = ""

with open("medical/text.txt", "r", encoding="utf-8") as file:
    text = file.read()

REL = Namespace("http://relations.example.org/")
WD = Namespace("http://www.wikidata.org/wiki/")

# take entities and labels from the ontology
def extract_entities(ontology_file):
    g = Graph()
    g.parse(ontology_file, format="turtle")
   
    entities_list = []
    labels_list = []
    
    #entities with skos:prefLabel
    for s in g.subjects(RDF.type, None):
        pref_label = g.value(s, SKOS.prefLabel)
        if pref_label and isinstance(pref_label, Literal):
            entities_list.append(s)
            labels_list.append(pref_label.value)
   
    return entities_list, labels_list

# make a list of embeddings for each entity label
def get_ontology_embeddings(embeddings, entityLabels):
    ontologyEmbeddings = []
    print("Extraction des embeddings pour l'ontologie ...")
    for name in entityLabels:
        embedding = embeddings.embed_query(name)
        ontologyEmbeddings.append(embedding)

    # number of embeddings must match the number of labels
    if len(ontologyEmbeddings) != len(entityLabels):
        print("erreur nombre d'embeddings/labels")
    else:
        print("embeddings ontologie ok.")

    return ontologyEmbeddings




# from an embedding, find the closest entity in the ontology
def entityRetriever(embedding, ontologyEmbeddings,entities):
    best_similarity = -1
    best_entity = None
    for i, e in enumerate(ontologyEmbeddings):
        similarity = np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e))
        if similarity > best_similarity:
            best_similarity = similarity
            best_entity = entities[i]
    return best_entity, best_similarity

def chunk_text(text):
    # 1 chunk = 1 sentence
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    return chunks

def extract_key_phrases(doc, nlp):
    # ner
    entities = list(doc.ents)
    
    # take chunks with more than 1 word
    noun_chunks = [chunk for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    
    # add the chunks to the entities
    all_entities = entities + noun_chunks
    
    # remove duplicates
    unique_entities = []
    seen_texts = set()
    
    for ent in all_entities:
        normalized_text = ent.text.strip().lower()
        if normalized_text not in seen_texts and len(normalized_text) > 3:
            seen_texts.add(normalized_text)
            unique_entities.append(ent)
    
    return unique_entities

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
    DAO = FaissDAO(384)

    print("Extraction des entités et labels de l'ontologie ...")
    ontologyEntities, entityLabels = extract_entities(ontology_path)

    entityNames = [str(e).split("/")[-1] for e in ontologyEntities]
    print("entityNames (premiers 5):", entityNames[3001:3006] if len(entityNames) >= 5 else entityNames)
    print("entityNames size:", len(entityNames))
    print("entityLabels (premiers 5):", list(entityLabels)[3001:3006] if len(entityLabels) >= 5 else list(entityLabels))
    print("entityLabels size:", len(entityLabels))

    # check if the number of entities and labels match
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

    #replace spaces with _, remove special characters and convert to lowercase
    entityForURIRef = [re.sub(r'[^a-zA-Z0-9]', '_', ent.text.lower()) for ent in entities]
    
    print("Construction du graphe ...")
    g = Graph()
    
    ATC = Namespace("http://purl.bioontology.org/ontology/ATC/")
    g.bind("rel", REL)
    g.bind("atc", ATC)
    g.bind("wd", WD)
    
   
    #put chunks in the graph
    chunk_uris = []
    for i, chunk in enumerate(chunks):
        chunk_uri = URIRef(f"http://example.org/chunk_{i}")
        
        g.add((chunk_uri, RDF.type, URIRef("http://example.org/Chunk")))
        g.add((chunk_uri, REL.id, Literal(i)))
        g.add((chunk_uri, SKOS.prefLabel, Literal(chunk)))
        chunk_uris.append(chunk_uri)
    
    # Ajouter les entités du texte et les relier aux chunks
    #add entities from the text and link them to their chunks
    for i, ent in enumerate(entities):
        print("insertion des entités et leur embedding dans le DAO")
        DAO.insert(ent.text, embeddings.embed_query(ent.text))
        #make an uri
        entity_uri = URIRef(f"http://example.org/entity/{entityForURIRef[i]}")
        
        ontology_entity, similarity = entityRetriever(embeddings.embed_query(ent.text), ontologyEmbeddings,ontologyEntities)

        g.add((entity_uri, SKOS.prefLabel, Literal(ent.text)))
        
        #if the similarity with the ontology is sufficient, link to the ontology
        if similarity >= 0.5:
            g.add((entity_uri, RDF.type, URIRef(ontology_entity)))
            g.add((entity_uri, REL.alignmentScore, Literal(similarity)))
            g.add((entity_uri, REL.alignedWith, URIRef(ontology_entity)))
            print(f"Entity '{ent.text}' aligned with ATC: {ontology_entity} (score: {similarity:.4f})")
        else:
            # wikidata_entity = wikidatautils.get_uri_wikidata(ent.text) 
            # g.add((entity_uri, RDF.type, URIRef(wikidata_entity)))
            # wikidatalabel = wikidatautils.get_description_from_entity(wikidata_entity)
            # print("label : ", wikidatalabel)
            # g.add((entity_uri, SKOS.prefLabel, Literal(wikidatalabel)))
            # wikidataLabelList.append(wikidatalabel)
            print(f"Entity '{ent.text}' not aligned with ATC")
        
        # find the chunks that contain this entity and link them
        chunk_indices = find_entity_in_chunks(ent.text, chunks)
        for idx in chunk_indices:
            g.add((entity_uri, REL.mentionedIn, chunk_uris[idx]))
            g.add((chunk_uris[idx], REL.mentions, entity_uri))

   
    g.serialize(destination=rdf_path, format="turtle")
    #use entityLinker to link entities to wikidata
    entityLinker.add_entity_linked_to_graph(rdf_path, "medical/outputLinker.ttl", text)
    wiki_label_list = entityLinker.link_wikiData_entities_to_chunks("medical/outputLinker.ttl", "medical/outputLinkerLinked.ttl")

    for label in wiki_label_list:
        if label not in wikidataLabelList:
            wikidataLabelList.append(label)
            DAO.insert(label, embeddings.embed_query(label))






    

    print(f"Graphe sauvegardé, {len(g)} triplets.")

    DAO.save_index("medical/embeddings.index")
    convert_wikidata_with_regex("medical/outputLinkerLinked.ttl", "medical/outputLinkerLinked.ttl")
    #add neughbors to the graph
    wikidatautils.add_wikidata_neighbors_to_graph("medical/outputLinkerLinked.ttl", output_path="medical/outputLinkerLinked.ttl" )  

    relations_dictionary = wikidatautils.verbalize_rdf_relations("medical/outputLinkerLinked.ttl") # a refaire 

    DAO_relations = FaissDAO_relations(384)
    for relation in relations_dictionary:
        source = relation['source']
        destination = relation['destination']
        verbalization = relation['verbalization']
        DAO_relations.insert(verbalization, source, destination, embeddings.embed_query(verbalization))


    DAO_relations.save_index("medical/relations.index")
    print(f"Relations sauvegardées dans 'medical/relations.index' avec {len(relations_dictionary)} relations.")
    wikidatautils.align_unlinked_entities_to_wikidata("medical/outputLinkerLinked.ttl", "medical/outputLinkerLinked.ttl")

    return g

def convert_wikidata_with_regex(input_file, output_file):

  
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    

    pattern = r'wd:Q(\d+)(?![a-zA-Z])'
    

    transformed = re.sub(pattern, r'<https://www.wikidata.org/wiki/Q\1>', content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transformed)
    
    print(f"Conversion terminée : {output_file} créé avec succès.")




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

def process_query(query_text, rdf_graph_path, embedding_model=embeddings, output_file="medical/query_enrichie.txt", neighborChunks=0):
    verbalisation_list = []
    DAO = FaissDAO(384)
    DAO.load_index("medical/embeddings.index")
    DAI_relations = FaissDAO_relations(384)
    DAI_relations.load_index("medical/relations.index")
    
    g = Graph()
    g.parse(rdf_graph_path, format="turtle")
    g.bind("rel", REL)
    
    #extraire les entités de la query
    doc = nlp(query_text)

    print("Extraction des entités de la requête ...")
    t2  = time.time()
    query_entities = extract_key_phrases(doc, nlp)
    print("temps d'extraction des entités de la requête : ", time.time()-t2)

    enriched_results = ["question :","\n\n", query_text, "\n\n"]
    enriched_context = []
    enriched_neighboor = []
    enriched_context.append("context : ")
    enriched_context.append("\n\n")
    enriched_verbalisation = []
    enriched_verbalisation.append("detail of entities detected in the query : \n\n")
    enriched_verbalisation.append("\n\n")
    enriched_neighboor.append("relations with other entities : \n\n")
    enriched_neighboor.append("\n\n")


    print("extraction des labels du graphe ...")
    t3 = time.time()
    labelsList = extract_labels_from_graph(g)
    print("temps d'extraction des labels : ", time.time()-t3)
    #print("labelsList : ", labelsList)

    # embeddingsList = get_embeddings_from_labels(labelsList)

    #print ("embeddingsList : ", embeddingsList)
    chunks_already_mentioned = set()
    chunkList = []

    print("initialisation du fichier json ...")
    t4 = time.time()
    wikidatautils.initialize_json_file("medical/logs.json")
    print("temps d'initialisation du fichier json : ", time.time()-t4)
    
    print("dernière partie ...")
    t5 = time.time()

    #neighbor_relations, _ = DAI_relations.search(embeddings.embed_query(query_text), k=5)
    #print("neighbor_relations : ", neighbor_relations)






    #pour chaque entité dans la requete
    for ent in query_entities:
        print (f"ent : {ent.text}")

        # correspondingEnt = retrieve_corresponding_label(ent, embeddingsList, labelsList)
        correspondingEnt, distance = DAO.search(embeddings.embed_query(ent.text), k=1)
        print("correspondingEnt de : ",ent.text," ", correspondingEnt)
        verbalisation = wikidatautils.analyze_entity(rdf_graph_path,correspondingEnt[0],"medical/ATC.ttl")
        print("verbalisation : ",verbalisation )
        verbalisation_list.append(verbalisation)
        enriched_verbalisation.append(verbalisation)
        neighbor_relations_list = []

        l, chunkList = wikidatautils.retrieve_mentioned_chunks(rdf_graph_path, correspondingEnt[0], chunkList, neighborChunks)
        neighbor_relations, _ = DAI_relations.search_specific(embeddings.embed_query(query_text),correspondingEnt[0], k=5)
        for relation in neighbor_relations:
            source = relation['source']
            destination = relation['destination']
            relation = relation['label']
            # # if the entity corresponds or is contained in to the source or destination, add the verbalisation (no case sensistive)
            # if correspondingEnt[0].lower() in source.lower()  or correspondingEnt[0].lower() in destination.lower():
            # # if source.lower() == correspondingEnt[0].lower() or destination.lower() == correspondingEnt[0].lower():

            neighbor_relations_list.append(f"{source} {relation} {destination}")
            print("verbalisation de la relation : ", relation)        
            enriched_neighboor.append(f"{source} {relation} {destination}")

        
        entity_data = {
            'entity': ent.text,
            'correspondingEnt': correspondingEnt,
            'chunkCount': l  ,
            'verbalisation': verbalisation,
            'neighborRelations': neighbor_relations_list
        }
        wikidatautils.add_to_json_file("medical/logs.json", entity_data)

    for chunk in chunkList:
        enriched_context.append(chunk)
    print("chunks_already_mentioned : ", chunks_already_mentioned)

    enriched_results.append("\n".join(enriched_verbalisation))
    enriched_results.append("\n")
    enriched_results.append("\n".join(enriched_neighboor))
    enriched_results.append("\n")
    enriched_results.append("\n".join(enriched_context))

    

    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(enriched_results))
    
    print("temps de la dernière partie : ", time.time()-t5)

    print(f"Requête enrichie sauvegardée dans {output_file}")
    print("total des verbalisations : ", verbalisation_list)
    print("5 meilleures realtions avec les voisins : ", neighbor_relations)
    return "\n".join(enriched_results)

# à commenter pour pas reconstruire le graphe
# build_knowledge_graph_aligned_with_ontology(text, "medical/ATC.ttl", nlp, "medical/knowledge_graphNoWiki.ttl", embeddings)
# build_knowledge_graph_aligned_with_ontology(text, "medical/ATC.ttl", nlp, "medical/knowledge_graphNoWiki.ttl", embeddings)

print ("enrichissement de la requête ...")

# process_query("cell-specific targeting", "medical/outputLinkerLinked.ttl", embeddings)
# process_query("What is the amino acid similarity between IFITM5 and the other IFITM proteins? ","medical/outputLinkerLinked.ttl", embeddings) 
# process_query("What is the main cause of hiv infection on children? hiv hiv hiv hiv ","medical/outputLinkerLinked.ttl", embeddings) 
# process_query("Which Human Coronavirus showed species specific clinical characteristics of its infection?","medical/outputLinkerLinked.ttl", embeddings) 


