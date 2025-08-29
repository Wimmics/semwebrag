import configparser
import logging
import os
from pathlib import Path
from rdflib import Graph, RDF, RDFS, Literal, URIRef, Namespace
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
import shutil

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.ini"):
    """Load configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def setup_domain_directory(domain):
    """Create domain directory if it doesn't exist"""
    domain_path = Path(domain)
    domain_path.mkdir(exist_ok=True)
    logger.info(f"Domain directory created/verified: {domain_path}")
    return domain_path

def initialize_components(config):
    """Initialize embeddings, nlp model and other components"""
    # Load embeddings
    local_model_path = config['EMBEDDINGS']['local_model_path']
    embeddings = HuggingFaceEmbeddings(model_name=local_model_path)
    logger.info(f"Embeddings loaded from: {local_model_path}")
    
    # Load spacy model
    spacy_model = config['NLP']['spacy_model']
    nlp = spacy.load(spacy_model)
    logger.info(f"SpaCy model loaded: {spacy_model}")
    
    return embeddings, nlp

def load_text_data(text_file_path):
    """Load text from file"""
    try:
        with open(text_file_path, "r", encoding="utf-8") as file:
            text = file.read()
        logger.info(f"Text loaded from: {text_file_path}")
        return text
    except FileNotFoundError:
        logger.error(f"Text file not found: {text_file_path}")
        raise

def extract_entities(ontology_file):
    """Extract entities and labels from ontology, supporting both SKOS.prefLabel and RDFS.label"""
    logger.info("Extracting entities and labels from ontology...")
    g = Graph()
    g.parse(ontology_file, format="turtle")
   
    entities_list = []
    labels_list = []
    
    for s in g.subjects(RDF.type, None):
        # Try SKOS.prefLabel first
        pref_label = g.value(s, SKOS.prefLabel)
        if pref_label and isinstance(pref_label, Literal):
            entities_list.append(s)
            labels_list.append(pref_label.value)
        else:
            # If no SKOS.prefLabel, try RDFS.label
            rdfs_label = g.value(s, RDFS.label)
            if rdfs_label and isinstance(rdfs_label, Literal):
                entities_list.append(s)
                labels_list.append(rdfs_label.value)
   
    logger.info(f"Entities extracted: {len(entities_list)}, Labels: {len(labels_list)}")
    return entities_list, labels_list

def get_ontology_embeddings(embeddings, entityLabels):
    ontologyEmbeddings = []
    logger.info("Extracting embeddings for ontology...")
    
    for name in entityLabels:
        embedding = embeddings.embed_query(name)
        ontologyEmbeddings.append(embedding)

    if len(ontologyEmbeddings) != len(entityLabels):
        logger.error("Error: number of embeddings/labels doesn't match")
        raise ValueError("Number of embeddings doesn't match number of labels")
    else:
        logger.info("Ontology embeddings generated successfully")

    return ontologyEmbeddings

def entityRetriever(embedding, ontologyEmbeddings, entities):
    """Find the closest entity in ontology from an embedding"""
    best_similarity = -1
    best_entity = None
    for i, e in enumerate(ontologyEmbeddings):
        similarity = np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e))
        if similarity > best_similarity:
            best_similarity = similarity
            best_entity = entities[i]
    return best_entity, best_similarity

def chunk_text(text, nlp):
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    logger.info(f"Text divided into {len(chunks)} chunks")
    return chunks

def extract_key_phrases(doc, nlp):
    entities = list(doc.ents)
    noun_chunks = [chunk for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    all_entities = entities + noun_chunks
    
    unique_entities = []
    seen_texts = set()
    
    for ent in all_entities:
        normalized_text = ent.text.strip().lower()
        if normalized_text not in seen_texts and len(normalized_text) > 3:
            seen_texts.add(normalized_text)
            unique_entities.append(ent)
    
    logger.info(f"Unique entities extracted: {len(unique_entities)}")
    return unique_entities

def find_entity_in_chunks(entity_text, chunks):
    """Find chunks that contain a given entity"""
    entity_text_lower = entity_text.lower()
    chunk_indices = []
    
    for i, chunk in enumerate(chunks):
        if entity_text_lower in chunk.lower():
            chunk_indices.append(i)
    
    return chunk_indices

def build_knowledge_graph_aligned_with_ontology(text, ontology_path, nlp, rdf_path, embeddings, domain_path, embedding_dimensions):
    """Build knowledge graph aligned with ontology"""
    wikidataLabelList = []
    doc = nlp(text)
    chunks = chunk_text(text, nlp)
    
    # Create DAO with specified dimensions
    DAO = FaissDAO(embedding_dimensions)

    logger.info("Extracting entities and labels from ontology...")
    ontologyEntities, entityLabels = extract_entities(ontology_path)

    entityNames = [str(e).split("/")[-1] for e in ontologyEntities]
    logger.info(f"EntityNames - first 5: {entityNames[3001:3006] if len(entityNames) >= 5 else entityNames}")
    logger.info(f"EntityNames size: {len(entityNames)}")
    logger.info(f"EntityLabels - first 5: {list(entityLabels)[3001:3006] if len(entityLabels) >= 5 else list(entityLabels)}")
    logger.info(f"EntityLabels size: {len(entityLabels)}")

    if len(ontologyEntities) != len(entityLabels):
        logger.error("Error: number of entities/number of labels doesn't match")
        raise ValueError("Number of entities doesn't match number of labels")
    else:
        logger.info("Number of entities/labels is identical")

    ontologyEmbeddings = get_ontology_embeddings(embeddings, entityLabels)
    
    logger.info(f"Number of entities: {len(ontologyEntities)}")
    logger.info(f"Number of labels: {len(entityLabels)}")
    logger.info(f"Number of embeddings: {len(ontologyEmbeddings)}")

    # NER
    entities = extract_key_phrases(doc, nlp)
        
    logger.info("Text entities:")
    for ent in entities:
        logger.info(f"- {ent.text} ({ent.label_ if hasattr(ent, 'label_') else 'PHRASE'})")

    entityForURIRef = [re.sub(r'[^a-zA-Z0-9]', '_', ent.text.lower()) for ent in entities]
    
    logger.info("Building graph...")
    g = Graph()
    
    REL = Namespace("http://relations.example.org/")
    WD = Namespace("http://www.wikidata.org/wiki/")
    ATC = Namespace("http://purl.bioontology.org/ontology/ATC/")
    
    g.bind("rel", REL)
    g.bind("atc", ATC)
    g.bind("wd", WD)
    
    # Add chunks to graph
    chunk_uris = []
    for i, chunk in enumerate(chunks):
        chunk_uri = URIRef(f"http://example.org/chunk_{i}")
        
        g.add((chunk_uri, RDF.type, URIRef("http://example.org/Chunk")))
        g.add((chunk_uri, REL.id, Literal(i)))
        g.add((chunk_uri, SKOS.prefLabel, Literal(chunk)))
        chunk_uris.append(chunk_uri)
    
    # Add text entities and link them to chunks
    for i, ent in enumerate(entities):
        logger.info("Inserting entities and their embedding into DAO")
        DAO.insert(ent.text, embeddings.embed_query(ent.text))
        entity_uri = URIRef(f"http://example.org/entity/{entityForURIRef[i]}")
        
        ontology_entity, similarity = entityRetriever(embeddings.embed_query(ent.text), ontologyEmbeddings, ontologyEntities)

        g.add((entity_uri, SKOS.prefLabel, Literal(ent.text)))
        
        # If similarity with ontology is sufficient, link to ontology
        alignment_threshold = 0.5  # Could be in config
        if similarity >= alignment_threshold:
            g.add((entity_uri, RDF.type, URIRef(ontology_entity)))
            g.add((entity_uri, REL.alignmentScore, Literal(similarity)))
            g.add((entity_uri, REL.alignedWith, URIRef(ontology_entity)))
            logger.info(f"Entity '{ent.text}' aligned with ATC: {ontology_entity} (score: {similarity:.4f})")
        else:
            logger.info(f"Entity '{ent.text}' not aligned with ATC")
        
        # Find chunks that contain this entity and link them
        chunk_indices = find_entity_in_chunks(ent.text, chunks)
        for idx in chunk_indices:
            g.add((entity_uri, REL.mentionedIn, chunk_uris[idx]))
            g.add((chunk_uris[idx], REL.mentions, entity_uri))

    g.serialize(destination=rdf_path, format="turtle")
    
    # Paths for temporary files in domain
    output_linker_path = domain_path / "outputLinker.ttl"
    output_linker_linked_path = domain_path / "outputLinkerLinked.ttl"
    
    # Use entityLinker to link entities to wikidata
    entityLinker.add_entity_linked_to_graph(rdf_path, str(output_linker_path), text)
    wiki_label_list = entityLinker.link_wikiData_entities_to_chunks(str(output_linker_path), str(output_linker_linked_path))

    for label in wiki_label_list:
        if label not in wikidataLabelList:
            wikidataLabelList.append(label)
            DAO.insert(label, embeddings.embed_query(label))

    logger.info(f"Graph saved with {len(g)} triples")

    # Save embeddings index in domain directory
    embeddings_index_path = domain_path / "embeddings.index"
    DAO.save_index(str(embeddings_index_path))
    
    convert_wikidata_with_regex(str(output_linker_linked_path), str(output_linker_linked_path))
    
    # Add neighbors to graph
    wikidatautils.add_wikidata_neighbors_to_graph(str(output_linker_linked_path), output_path=str(output_linker_linked_path))  

    relations_dictionary = wikidatautils.verbalize_rdf_relations(str(output_linker_linked_path))

    DAO_relations = FaissDAO_relations(embedding_dimensions)
    for relation in relations_dictionary:
        source = relation['source']
        destination = relation['destination']
        verbalization = relation['verbalization']
        DAO_relations.insert(verbalization, source, destination, embeddings.embed_query(verbalization))

    # Save relations index in domain directory
    relations_index_path = domain_path / "relations.index"
    DAO_relations.save_index(str(relations_index_path))
    logger.info(f"Relations saved in '{relations_index_path}' with {len(relations_dictionary)} relations")
    
    wikidatautils.align_unlinked_entities_to_wikidata(str(output_linker_linked_path), str(output_linker_linked_path))

    ontologie_copy_path = domain_path / "ontology.ttl"
    if not ontologie_copy_path.exists():
        shutil.copyfile(ontology_path, ontologie_copy_path)
        logger.info(f"Ontology copied to '{ontologie_copy_path}'")
    else:
        logger.info(f"Ontology copy already exists at '{ontologie_copy_path}'")
   

    return g

def convert_wikidata_with_regex(input_file, output_file):
    """Convert Wikidata references to complete URIs"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'wd:Q(\d+)(?![a-zA-Z])'
    transformed = re.sub(pattern, r'<https://www.wikidata.org/wiki/Q\1>', content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transformed)
    
    logger.info(f"Conversion completed: {output_file} created successfully")

def extract_labels_from_graph(graph):
    """Extract all labels from graph"""
    labelList = []
    for s, p, o in graph.triples((None, SKOS.prefLabel, None)):
        if isinstance(o, Literal):
            labelList.append(o.value)

    return labelList

def get_embeddings_from_labels(labelList, embeddings):
    """Generate embeddings from a list of labels"""
    embeddingsList = []
    for name in labelList:
        embedding = embeddings.embed_query(name)
        embeddingsList.append(embedding)

    return embeddingsList

def retrieve_corresponding_label(entity, embeddingsList, labelList, embeddings):
    """Retrieve the most similar corresponding label"""
    embedding = embeddings.embed_query(entity.text)
    best_similarity = -1
    best_entity = None
    for i, e in enumerate(embeddingsList):
        similarity = np.dot(embedding, e) / (np.linalg.norm(embedding) * np.linalg.norm(e))
        if similarity > best_similarity:
            best_similarity = similarity
            best_entity = labelList[i]

    return best_entity

def process_query(query_text, nlp_model, rdf_graph_path, domain_path, embeddings, embedding_dimensions, neighborChunks=0):
    """Process a query and enrich it with graph context"""
    verbalisation_list = []
    
    embeddings_index_path = f"{domain_path}/embeddings.index"# domain_path / "embeddings.index"
    relations_index_path = f"{domain_path}/relations.index"
    logs_path = f"{domain_path}/logs.json"
    query_output_path = f"{domain_path}/query_enrichie.txt"
    
    DAO = FaissDAO(embedding_dimensions)
    DAO.load_index(str(embeddings_index_path))
    DAI_relations = FaissDAO_relations(embedding_dimensions)
    DAI_relations.load_index(str(relations_index_path))
    
    g = Graph()
    g.parse(rdf_graph_path, format="turtle")
    REL = Namespace("http://relations.example.org/")
    g.bind("rel", REL)
    
    nlp = spacy.load(nlp_model)  
    doc = nlp(query_text)

    logger.info("Extracting query entities...")
    t2 = time.time()
    query_entities = extract_key_phrases(doc, nlp)
    logger.info(f"Query entity extraction time: {time.time()-t2:.2f}s")

    enriched_results = ["question :", "\n\n", query_text, "\n\n"]
    enriched_context = []
    enriched_neighboor = []
    enriched_context.append("context : ")
    enriched_context.append("\n\n")
    enriched_verbalisation = []
    enriched_verbalisation.append("detail of entities detected in the query : \n\n")
    enriched_verbalisation.append("\n\n")
    enriched_neighboor.append("relations with other entities : \n\n")
    enriched_neighboor.append("\n\n")

    logger.info("Extracting labels from graph...")
    t3 = time.time()
    labelsList = extract_labels_from_graph(g)
    logger.info(f"Label extraction time: {time.time()-t3:.2f}s")

    chunks_already_mentioned = set()
    chunkList = []

    logger.info("Initializing JSON file...")
    t4 = time.time()
    wikidatautils.initialize_json_file(str(logs_path))
    logger.info(f"JSON file initialization time: {time.time()-t4:.2f}s")
    
    logger.info("Processing entities...")
    t5 = time.time()


    neighbor_relations = []
    
    # For each entity in the query
    for ent in query_entities:
        logger.info(f"Processing entity: {ent.text}")

        correspondingEnt, distance = DAO.search(embeddings.embed_query(ent.text), k=1)
        logger.info(f"Corresponding entity for {ent.text}: {correspondingEnt}")
        
        # Use domain ontology
        ontology_path = f"{domain_path}/ontology.ttl"  
        # logger.info(f"ONTOLOGY PATH: {ontology_path}")
        verbalisation = wikidatautils.analyze_entity(rdf_graph_path, correspondingEnt[0], str(ontology_path))
        logger.info(f"Verbalization: {verbalisation}")
        verbalisation_list.append(verbalisation)
        enriched_verbalisation.append(verbalisation)
        neighbor_relations_list = []

        l, chunkList = wikidatautils.retrieve_mentioned_chunks(rdf_graph_path, correspondingEnt[0], chunkList, neighborChunks)
        neighbor_relations, _ = DAI_relations.search_specific(embeddings.embed_query(query_text), correspondingEnt[0], k=5)
        
        for relation in neighbor_relations:
            source = relation['source']
            destination = relation['destination']
            relation_text = relation['label']

            neighbor_relations_list.append(f"{source} {relation_text} {destination}")
            logger.info(f"Relation verbalization: {relation_text}")        
            enriched_neighboor.append(f"{source} {relation_text} {destination}")
        
        entity_data = {
            'entity': ent.text,
            'correspondingEnt': correspondingEnt,
            'chunkCount': l,
            'verbalisation': verbalisation,
            'neighborRelations': neighbor_relations_list
        }
        wikidatautils.add_to_json_file(str(logs_path), entity_data)

    for chunk in chunkList:
        enriched_context.append(chunk)
    logger.info(f"Chunks already mentioned: {chunks_already_mentioned}")

    enriched_results.append("\n".join(enriched_verbalisation))
    enriched_results.append("\n")
    enriched_results.append("\n".join(enriched_neighboor))
    enriched_results.append("\n")
    enriched_results.append("\n".join(enriched_context))
    
    with open(query_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(enriched_results))
    
    logger.info(f"Entity processing time: {time.time()-t5:.2f}s")
    logger.info(f"Enriched query saved in {query_output_path}")
    logger.info(f"Total verbalizations: {verbalisation_list}")
    logger.info(f"Top 5 neighbor relations: {neighbor_relations}")
    
    return "\n".join(enriched_results)



def process_query_classic_RAG(query_text, nlp_model, rdf_graph_path, domain_path, embeddings, embedding_dimensions, neighborChunks=0):
    nlp = spacy.load(nlp_model)
    doc = nlp(query_text)
    tload = time.time()
    DAO = FaissDAO(384)
    DAO.load_index(f"{domain_path}Classic/embeddings.index")
    chunks_already_mentioned = []
    tloaded = time.time()-tload
    output_file = f"{domain_path}Classic/query_enrichie.txt"
    
    #extraire les entités de la query
    textract = time.time()


    enriched_results = ["question :","\n\n", query_text, "\n\n"]
    enriched_results.append("context : ")
    enriched_results.append("\n\n")

 
    
    textracted = time.time()-textract

    logger.info("dernière partie ...")
    t = time.time()
    wikidatautils.initialize_json_file(f"{domain_path}Classic/logs.json")
    #pour chaque entité dans la requete
    embedded_query = embeddings.embed_query(query_text)
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
        logger.info(f"Entité {i+1} : {ent} (distance : ")#{distance[i]})")
        if ent not in chunks_already_mentioned:
            enriched_results.append(f"{ent}\n")
            chunks_already_mentioned.append(ent)

    entity_data = {
        'entity': query_text,
        'correspondingEnt': correspondingEnt,
        'chunkCount': correspondingEnt  
        }
    wikidatautils.add_to_json_file(f"{domain_path}Classic/logs.json", entity_data)


    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(enriched_results))

    logger.info("derniere partie finie en : ", time.time()-t, "s")


    logger.info("temps chargment : ", tload, "s")
    logger.info("temps extraction : ", textract, "s")
    
    logger.info(f"Requête enrichie sauvegardée dans {output_file}")
    return "\n".join(enriched_results)


def main():

    config = load_config()
    
    domain = config['GENERAL']['domain']
    text_file_path = config['DATA']['text_file_path']
    ontology_path = config['DATA']['ontology_path']
    embedding_dimensions = config.getint('EMBEDDINGS', 'dimensions')
    spacy_model = config['NLP']['spacy_model']
    
    domain_path = setup_domain_directory(domain)


    
    embeddings, nlp = initialize_components(config)
    
    text = load_text_data(text_file_path)
    
    rdf_path = domain_path / "knowledge_graph.ttl"
    
    # build_knowledge_graph_aligned_with_ontology(
    #     text, ontology_path, nlp, str(rdf_path), embeddings, domain_path, embedding_dimensions
    # )
    
    logger.info("Query enrichment...")
    
    
    # process_query("what are NVIDIA's main domains of activities ?", str(domain_path / "outputLinkerLinked.ttl"), domain_path, embeddings, embedding_dimensions)

if __name__ == "__main__":
    main()