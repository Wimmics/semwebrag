from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
#import finance.pipelineLinkerF
import os
from sys import argv
# from langchain.embeddings import HuggingFaceEmbeddings
import datetime
import re
import time
# from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import spacy
import importlib
import random
import requests



venv_path = 'venv'
domain = 'finance'
key = ''
with open("key.txt", "r", encoding="utf-8") as file:
    key = file.read()


def chat_completion(question: str):
    with open("key.txt", "r", encoding="utf-8") as file:
        key = file.read().strip()

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
 
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": "llama3-70b-8192",  
        "messages": [
            {"role": "system", "content": "You are Nestor, a virtual assistant. Answer to the question by using the context given bellow."},
            {"role": "user", "content": question}
        ],
        "max_tokens": 1500
    }
    



    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f" {content}")
        return content
    else:
        error_message = f"Erreur: {response.status_code} - {response.text}"
        print(error_message)
        return error_message




def readLog(text_path) : 
    res = ''
    with open (text_path, "r", encoding="utf-8") as file:
        res = file.read()

    return res

def run_in_venv_query(query, domain, nChunks=0) :
         # Chemin vers le Python de l'environnement virtuel sous Windows
    venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    print ("run_in_venv_query domain : ", domain)
    
    try:
        result = subprocess.run([venv_python, '-m', f'{domain}.userPrompt', query, str(nChunks)], 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)  
    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'exécution:")
        print(e.stderr)
        print(f"Code de retour : {e.returncode}")
        return 'Erreur lors de l\'exécution prompt'




app = Flask(__name__)

# Activer CORS pour toutes les origines
CORS(app)
# CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/ask', methods=['POST'])
def ask():
    # Récupérer la question envoyée par l'utilisateur
    data = request.get_json()
    user_prompt = data.get('prompt')
    domain = data.get('domain')
    nChunks = data.get('nChunks')

    print("domain : ",domain)
    print ("prompt : ", user_prompt)

    finance_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), domain)
    
    t=time.time()
    run_in_venv_query(user_prompt, domain, nChunks)
    print ("temps d'execution de la requete : ", time.time()-t)

    # Appeler le script Python avec le prompt
    
    try:

        t = time.time()
        
        prompt = ""
        with open (f'{domain}/query_enrichie.txt', "r", encoding="utf-8") as file:
            prompt = file.read()
        response = chat_completion(prompt)
        print ("temps d'execution du script 2 : ", time.time()-t)
        
        return jsonify({'response': response})
    except subprocess.CalledProcessError as e:
        return jsonify({'response': 'Erreur dans l\'exécution du script Python.'}), 500

    

@app.route('/log', methods=['POST'])  
def log():
    data = request.get_json()
    user_prompt = data.get('prompt')
    domain = data.get('domain')

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return jsonify({'response': f'{date} <br><br> Question : {user_prompt} <br><br> log : {readLog(f'{domain}/log.txt')} '})

@app.route('/currentTime', methods=['GET'])
def currentTime():
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'response': date})



@app.route('/evaluate', methods=['GET'])
def evaluate():
    venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    try:

        domain = request.args.get('domain')
        nChunks = request.args.get('nChunks')
        question = request.args.get('question')
        true_answer = request.args.get('true_answer')

        
        result = subprocess.run(
            [venv_python, '-m', f'{domain}.eval', nChunks, question, true_answer],
            capture_output=True,
            text=True,
            check=True
        )
        

        output = result.stdout
        
        question_match = re.search(r"Question : (.*)", output)
        true_answer_match = re.search(r"True Answer : (.*)", output)
        predicted_answer_match = re.search(r"Predicted Answer : (.*)", output)
        meteor_match = re.search(r"(METEOR Score:.*)", output)
        bleu_match = re.search(r"(BLEU Score:.*)", output)
        bert_match = re.search(r"(BERT Score Precision:.*)", output)
        rouge_match = re.search(r"(rougeScores:.*)", output)
        overlap_match = re.search(r"(Overlap Coefficient:.*)", output)
        overlapE_match = re.search(r"(OverlapE Coefficient:.*)", output)
        
        # Extraire les valeurs (ou utiliser des chaînes vides si non trouvées)
        question = question_match.group(1).strip() if question_match else ""
        true_answer = true_answer_match.group(1).strip() if true_answer_match else ""
        predicted_answer = predicted_answer_match.group(1).strip() if predicted_answer_match else ""
        meteor = meteor_match.group(1).strip() if meteor_match else ""
        bleu = bleu_match.group(1).strip() if bleu_match else ""
        bert = bert_match.group(1).strip() if bert_match else ""
        rouge = rouge_match.group(1).strip() if rouge_match else ""
        overlap = overlap_match.group(1).strip() if overlap_match else ""
        overlapE = overlapE_match.group(1).strip() if overlapE_match else ""
        
       
        response_data = {
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "meteor": meteor,
            "bleu": bleu,
            "bert": bert,
            "rouge": rouge,
            "overlap": overlap,
            "overlapE": overlapE
        }
        print("meteor : ", meteor)
        print("bleu : ", bleu)
        print("bert : ", bert)
        print ("rouge : ", rouge)
        print ("overlap : ", overlap)
        print ("overlapE : ", overlapE)
        
        return jsonify(response_data)
    
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Erreur lors de l'exécution du script d'évaluation",
            "details": e.stderr,
            "return_code": e.returncode
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Erreur inattendue",
            "details": str(e)
        }), 500


@app.route('/compare', methods=['GET'])
def compare():
    venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    try:
        # Récupérer les paramètres de la requête
        question = request.args.get('question', '')
        true_answer = request.args.get('true_answer', '')
        domain = request.args.get('domain')
        
        if not question or not true_answer:
            return jsonify({
                "error": "Les paramètres 'question' et 'true_answer' sont requis"
            }), 400
        

        domain = "finance"  # À ajuster selon votre structure de projet
        
        result = subprocess.run(
            [venv_python, '-m', f'{domain}.compare', question, true_answer],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Récupérer la sortie
        output = result.stdout
        
        # Analyser la sortie pour extraire les informations
        predicted_answer_match = re.search(r"Predicted Answer : (.*)", output)
        meteor_match = re.search(r"(METEOR Score:.*)", output)
        bleu_match = re.search(r"(BLEU Score:.*)", output)
        bert_match = re.search(r"(BERT Score Precision:.*)", output)
        rouge_match = re.search(r"(rougeScores:.*)", output)
        
        # Extraire les valeurs (ou utiliser des chaînes vides si non trouvées)
        predicted_answer = predicted_answer_match.group(1).strip() if predicted_answer_match else ""
        meteor = meteor_match.group(1).strip() if meteor_match else ""
        bleu = bleu_match.group(1).strip() if bleu_match else ""
        bert = bert_match.group(1).strip() if bert_match else ""
        rouge = rouge_match.group(1).strip() if rouge_match else ""
        
        # Préparer les données à retourner
        response_data = {
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "meteor": meteor,
            "bleu": bleu,
            "bert": bert,
            "rouge": rouge
        }
        
        return jsonify(response_data)
    
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Erreur lors de l'exécution du script de comparaison",
            "details": e.stderr,
            "return_code": e.returncode
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Erreur inattendue",
            "details": str(e)
        }), 500


nlp = spacy.load("en_core_web_md")

nlpfr = spacy.load("fr_core_news_md")


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



@app.route('/responseEntityDetection', methods=['POST'])
def responseEntityDetection():
    # venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    #return ["the", "in", "or", "a"]
    data = request.get_json()
    rep = data.get('response')
    doc = nlp(rep)
    res = extract_key_phrases(doc, nlp)
    return jsonify({ 'entities': [ent.text for ent in res] })
    # return jsonify({ 'entities': ["the", "in", "or", "a"] })


@app.route('/getAllQA', methods=['GET'])
def getAllQA():

    domain = request.args.get('domain')
    module = importlib.import_module(f'{domain}.getAllQA')
    data = module.getAllQA()
   
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)



