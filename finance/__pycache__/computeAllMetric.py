# prendre toutes les questions (getAllQA), les évaluer et stocker la moyenne de chaque métrique dans un fichier csv

import csv
import os
import code
import subprocess
import time
import random
import nltk
import pandas as pd
from finance.getAllQA import getAllQA
import json
import re
import time
from time import sleep


allQAJson = getAllQA()
questions = allQAJson["questions"]
answers = allQAJson["answers"]

nChunk = 0 

venv_python = os.path.join('venv', 'Scripts', 'python.exe')


#créer un fichier csv avec les colonnes : nchunks, meteor, bleu, bert, rouge

with open("finance/evaluation.csv", mode="w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "nChunks",
        "METEOR", "BLEU", "BERT", "ROUGE"
    ])
    writer.writeheader()
    for i in range(11):  # nChunks de 0 à 10
        writer.writerow({
            "nChunks": i,
            "METEOR": 0,
            "BLEU": 0,
            "BERT": 0,
            "ROUGE": 0
        })

print("csv créé")

#fonction pour ajouter les valeurs de chaque métrique à la ligne correspondante de nChunks dans le fichier csv
def addToCSV(nChunks, meteor, bleu, bert, rouge):
    df = pd.read_csv('finance/evaluation.csv')
    df.loc[df['nChunks'] == nChunks, 'METEOR'] = meteor
    df.loc[df['nChunks'] == nChunks, 'BLEU'] = bleu
    df.loc[df['nChunks'] == nChunks, 'BERT'] = bert
    df.loc[df['nChunks'] == nChunks, 'ROUGE'] = rouge
    df.to_csv('finance/evaluation.csv', index=False)


while(nChunk<10):
    resMeteor=0
    resBleu=0
    resBert=0
    resRouge=0
    for question in questions:
        print("index : ", questions.index(question))
        answer = answers[questions.index(question)]
        # if(questions.index(question) > 11):
        #     break
        result = subprocess.run([venv_python, '-m', 'finance.eval', str(nChunk), question, answer], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)

        output = (result.stdout)
        print("output : ", output)
        # meteor_match = re.search(r"(METEOR Score:.*)", output)
        meteor_match = re.search(r"(METEOR Score: ([\d.]+))", output)

        resM = float(meteor_match.group(2)) if meteor_match else None
        print("resM : ", resM)
        resMeteor+=resM
        # bleu_match = re.search(r"(BLEU Score:.*)", output)
        bleu_match = re.search(r"(BLEU Score: ([\d.]+))", output)

        resBl = float(bleu_match.group(2)) if bleu_match else None
        print("resBl : ", resBl)
        resBleu+=resBl
        # bert_match = re.search(r"(BERT Score Precision:.*)", output)
        # rouge_match = re.search(r"(rougeScores:.*)", output)
        # rouge_match = re.search(r"(rougeL:.*)", output)
        # precision_match = re.search(r"precision=([\d.]+)", rougeL_text)
        bert_match = re.search(r"BERT Score Precision: ([\d.]+)", output)   
        print("bert_match : ",bert_match)
        resB = float(bert_match.group(1)) if bert_match else None
        
        print("resB : ", resB)
        rouge_match = re.search(r"rougeL: Score\(precision=([\d.]+)", output)
        resr = float(rouge_match.group(1)) if rouge_match else None
        print (resr)
        resRouge+=resr  
        resBert+=resB
        sleep(300)
        

    
    # print("resrouge : ", resRouge/len(questions))
    print("resMeteor : ", resMeteor/len(questions))
    print("resBleu : ", resBleu/len(questions))
    print("resBert : ", resBert/len(questions))
    print("resrouge : ", resRouge/len(questions))
    
    addToCSV(nChunk, resMeteor/len(questions), resBleu/len(questions), resBert/len(questions), resRouge/len(questions))


    nChunk += 1