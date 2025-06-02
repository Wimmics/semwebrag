
import nltk
from nltk.translate.meteor_score import meteor_score
#bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

from rouge_score import rouge_scorer

import csv

import random

import subprocess
import os
import time
from sys import argv
import spacy

def get_random_question_response(code="NVDA") : 
    #dans le fichier financ/Financial-QA-10k.csv, lire les lignes contenant le le code dans la colone "ticker", prendre une de ces lignes au hasard et renvoyer le contenu de la colonne question et answer
    with open("finance/Financial-QA-10k.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row["ticker"] == code]
        if rows:
            row = random.choice(rows)
            question = row["question"]
            answer = row["answer"]
            return question, answer

def get_all_questions_responses(code="NVDA") :
    #dans le fichier financ/Financial-QA-10k.csv, lire les lignes contenant le le code dans la colone "ticker", prendre une de ces lignes au hasard et renvoyer le contenu de la colonne question et answer
    with open("finance/Financial-QA-10k.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row["ticker"] == code]
        if rows:
            questions = []
            answers = []
            for row in rows:
                questions.append(row["question"])
                answers.append(row["answer"])
            return questions, answers


def calculate_overlap(set1, set2):

    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    smaller_set_size = min(len(set1), len(set2))
    
    if smaller_set_size == 0:
        return 0.0

    if len(set1) == 0 or len(set2) == 0:
        return 0.0
        
    # return intersection / smaller_set_size
    return intersection / len(set1)


def calculate_overlap_entities(true_answer, predicted_answer):
    true_entities = getEntities(true_answer)
    predicted_entities = getEntities(predicted_answer)

    print("True Entities:", [ent.text for ent in true_entities])
    print("Predicted Entities:", [ent.text for ent in predicted_entities])

    intersections = 0
    for true_entity in true_entities:
        for predicted_entity in predicted_entities:
            if true_entity.text.strip().lower() == predicted_entity.text.strip().lower():
                intersections += 1
                break  # Stop after finding the first match for this true entity
    print("Intersections:", intersections)

    return intersections /len(true_entities) if true_entities else 0.0
    #return calculate_overlap(set(true_entities), set(predicted_entities))

def getEntities(text):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    query_entities = extract_key_phrases(doc, nlp)
    return query_entities

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






def evaluate_metrics(nChunks, question, true_answer):

    nltk.download('punkt')
    nltk.download('wordnet')

    #question, true_answer = get_random_question_response()

    venv_python = os.path.join('venv', 'Scripts', 'python.exe')
    try:
        result = subprocess.run([venv_python, '-m', 'finance.userPrompt', question, nChunks], 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)  

    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'exécution:")
        print(e.stderr)
        print(f"Code de retour : {e.returncode}")
        return 'Erreur lors de l\'exécution prompt'

    time.sleep(1)

    try:

        pa = subprocess.run([venv_python, '-m', 'finance.prompt2'], capture_output=True, text=True, check=True)
        predicted_answer = pa.stdout
    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'exécution:")
        predicted_answer=''

    print("Question : ", question)

    print("Predicted Answer : ", predicted_answer)
    # print("\n")

    print("True Answer : ", true_answer)

    # print("\n")




    Mref = nltk.word_tokenize(true_answer)
    Mhyp = nltk.word_tokenize(predicted_answer)

    Mscore = meteor_score([Mref], Mhyp)

    print("METEOR Score:", Mscore)
    resMeteor = f"METEOR Score: {Mscore}"

    Bref = true_answer.split()
    Bhyp = predicted_answer.split()

    bleu_score = sentence_bleu([Bref], Bhyp)
    if bleu_score < 0.001:
        bleu_score = 0.0
    print("BLEU Score:", bleu_score)

    resBleiu = f"BLEU Score: {bleu_score}"


    BertRef = true_answer
    BertHyp = predicted_answer

    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([BertHyp], [BertRef])
    print(f"BERT Score Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    resBert = f"BERT Score Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}"

    rougerScorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rougeRef = true_answer
    rougeHyp = predicted_answer

    rougeScores = rougerScorer.score(rougeRef, rougeHyp)

    for key in rougeScores : 
        print(f'{key}: {rougeScores[key]}')

    resRouge = f"rougeScores: {rougeScores}"

    ref_tokens = set(nltk.word_tokenize(true_answer.lower()))
    hyp_tokens = set(nltk.word_tokenize(predicted_answer.lower()))
    
    overlap_coef = calculate_overlap(ref_tokens, hyp_tokens)
    resOverlap = f"Overlap Coefficient: {overlap_coef:.4f}"

    everlapE_coef = calculate_overlap_entities(true_answer, predicted_answer)
    resOverlapE = f"OverlapE Coefficient: {everlapE_coef:.4f}"

    print(resRouge)
    print(resOverlap)
    print(resOverlapE)
    print (resRouge)
    time.sleep(30)
    return question, true_answer, predicted_answer, resMeteor, resBleiu, resBert, resRouge, resOverlap, resOverlapE

evaluate_metrics(argv[1], argv[2], argv[3])

# print("overlape : ",calculate_overlap_entities(argv[1], argv[2]))