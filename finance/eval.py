
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

    print (resRouge)
    return question, true_answer, predicted_answer, resMeteor, resBleiu, resBert, resRouge

evaluate_metrics(argv[1], argv[2], argv[3])

