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

# Créer le fichier CSV avec les bonnes colonnes (ajout de la colonne TEMPS)
if not os.path.exists('finance/evaluation3.csv'):
    with open('finance/evaluation3.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['nChunks', 'METEOR', 'BLEU', 'BERT', 'ROUGE','OVERLAP','OVERLAPE', 'TEMPS'])

    # ajouter les lignes nchunk allant de 0 à 10, et les valeurs de chaque métrique à 0
    for i in range(11):
        with open('finance/evaluation3.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, 0, 0, 0, 0, 0, 0, 0])

#fonction pour ajouter les valeurs de chaque métrique à la ligne correspondante de nChunks dans le fichier csv
def addToCSV(nChunks, meteor, bleu, bert, rouge, overlap, overlapE, temps_moyen):
    df = pd.read_csv('finance/evaluation3.csv')
    
    # Vérifier si la ligne existe
    mask = df['nChunks'] == nChunks
    if mask.any():
        df.loc[mask, 'METEOR'] = meteor
        df.loc[mask, 'BLEU'] = bleu
        df.loc[mask, 'BERT'] = bert
        df.loc[mask, 'ROUGE'] = rouge
        df.loc[mask, 'OVERLAP'] = overlap
        df.loc[mask, 'OVERLAPE'] = overlapE
        df.loc[mask, 'TEMPS'] = temps_moyen
        df.to_csv('finance/evaluation3.csv', index=False)
        print(f"Données mises à jour pour nChunks = {nChunks}")
    else:
        print(f"Erreur: aucune ligne trouvée pour nChunks = {nChunks}")
        print(f"Valeurs disponibles dans nChunks: {df['nChunks'].unique()}")

nChunk = 0

while(nChunk <= 10):
    print(f"\n=== Traitement nChunk = {nChunk} ===")
    resMeteor = 0
    resBleu = 0
    resBert = 0
    resRouge = 0
    resOverlap = 0
    resOverlapE = 0
    temps_total = 0  # Pour stocker le temps total d'exécution
    nberror = 0
    
    for question in questions:
        print("index : ", questions.index(question))
        answer = answers[questions.index(question)]

            
        try:
            # Mesurer le temps de début
            start_time = time.time()
            
            result = subprocess.run([venv_python, '-m', 'finance.eval', str(nChunk), question, answer], 
                                      capture_output=True, 
                                      text=True, 
                                      check=True)
            
            # Mesurer le temps de fin
            end_time = time.time()
            execution_time = end_time - start_time

            output = (result.stdout)
            print("output : ", output)
            
            # Vérifier les erreurs
            if "Erreur: 413 - {" in output or "Erreur: 500 - {" in output or "Request too large for model" in output:
                print("Erreur détéctée")
                nberror += 1
            else:
                # Ajouter le temps d'exécution seulement si pas d'erreur
                temps_total += execution_time
                print(f"Temps d'exécution: {execution_time:.2f} secondes")
                
                # Extraction des métriques avec gestion d'erreur
                meteor_match = re.search(r"METEOR Score: ([\d.]+)", output)
                if meteor_match:
                    resM = float(meteor_match.group(1))
                    print("resM : ", resM)
                    resMeteor += resM
                else:
                    print("METEOR non trouvé")
                    nberror += 1
                    continue

                bleu_match = re.search(r"BLEU Score: ([\d.]+)", output)
                if bleu_match:
                    resBl = float(bleu_match.group(1))
                    print("resBl : ", resBl)
                    resBleu += resBl
                else:
                    print("BLEU non trouvé")
                    nberror += 1
                    continue

                bert_match = re.search(r"BERT Score Precision: ([\d.]+)", output)   
                if bert_match:
                    resB = float(bert_match.group(1))
                    print("resB : ", resB)
                    resBert += resB
                else:
                    print("BERT non trouvé")
                    nberror += 1
                    continue
                
                rouge_match = re.search(r"rougeL: Score\(precision=([\d.]+)", output)
                if rouge_match:
                    resr = float(rouge_match.group(1))
                    print("rouge : ", resr)
                    resRouge += resr
                else:
                    print("ROUGE non trouvé")
                    nberror += 1
                    continue
                
                overlap_match = re.search(r"Overlap Coefficient: ([\d.]+)", output)
                if overlap_match:
                    resO = float(overlap_match.group(1))
                    resOverlap += resO
                else:
                    print("Overlap non trouvé")
                    nberror += 1
                    continue
                    
                overlapE_match = re.search(r"OverlapE Coefficient: ([\d.]+)", output)
                if overlapE_match:
                    resOE = float(overlapE_match.group(1))
                    resOverlapE += resOE
                else:
                    print("OverlapE non trouvé")
                    nberror += 1
                    continue

        except subprocess.CalledProcessError as e:
            print(f"Erreur subprocess: {e}")
            nberror += 1
        except Exception as e:
            print(f"Erreur inattendue: {e}")
            nberror += 1

        sleep(50)
    
    # Calculer les moyennes en évitant la division par zéro
    nb_valid = len(quesitons) - nberror
    if nb_valid > 0:
        avg_meteor = resMeteor / nb_valid
        avg_bleu = resBleu / nb_valid if resBleu / nb_valid <= 1 else 0  # Cap BLEU à 1
        avg_bert = resBert / nb_valid
        avg_rouge = resRouge / nb_valid
        avg_overlap = resOverlap / nb_valid
        avg_overlapE = resOverlapE / nb_valid
        avg_temps = temps_total / nb_valid  # Temps moyen d'exécution
        
        print(f"resMeteor : {avg_meteor}")
        print(f"resBleu : {avg_bleu}")
        print(f"resBert : {avg_bert}")
        print(f"resrouge : {avg_rouge}")
        print(f"resOverlap : {avg_overlap}")
        print(f"resOverlapE : {avg_overlapE}")
        print(f"Temps moyen : {avg_temps:.2f} secondes")
        print(f"{nberror} erreurs détectées")

        addToCSV(nChunk, avg_meteor, avg_bleu, avg_bert, avg_rouge, avg_overlap, avg_overlapE, avg_temps)
    else:
        print(f"Aucune donnée valide pour nChunk = {nChunk}")
        addToCSV(nChunk, 0, 0, 0, 0, 0, 0, 0)

    nChunk += 1