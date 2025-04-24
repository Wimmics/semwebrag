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
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import spacy
import importlib
import random


venv_path = 'venv'
domain = 'finance'
key = ''
with open("key.txt", "r", encoding="utf-8") as file:
    key = file.read()

model = ChatMistralAI(model="Meta-Llama-3_1-70B-Instruct", 
                        api_key=key,
                        endpoint='https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1', 
                        max_tokens=1500)


def chat_completion(question: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Nestor, a virtual assistant. Answer to the question by using the context given bellow."),
        ("human", "{question}"),
    ])

    chain = prompt | model

    response = chain.invoke(question)

    print(f" {response.content}")
    return response.content


def readLog(text_path) : 
    res = ''
    with open (text_path, "r", encoding="utf-8") as file:
        res = file.read()

    return res

def run_in_venv_query(query, domain, nChunks=0) :
         # Chemin vers le Python de l'environnement virtuel sous Windows
    venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    
    try:
        result = subprocess.run([venv_python, '-m', f'{domain}.userPrompt', query, str(nChunks)], 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)  
        # print("Sortie standard:")
        # print(result.stdout)
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

    finance_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), domain)
    
    t=time.time()
    run_in_venv_query(user_prompt, domain, nChunks)
    print ("temps d'execution de la requete : ", time.time()-t)

    # Appeler le script Python avec le prompt
    
    try:
        # result = subprocess.run(['python', '-m', 'finance.prompt2'], capture_output=True, text=True, check=True)
        # response = result.stdout.strip()
        t = time.time()
        # response = run_in_venv_windows(domain)
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
    # Récupérer la question envoyée par l'utilisateur
    data = request.get_json()
    user_prompt = data.get('prompt')
    domain = data.get('domain')
    # finance_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), domain)
    # récuperer la date rpécise : 
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
        # Exécuter le module avec subprocess
        # Remplacez {domain} par le nom de votre domaine réel
        #domain = "finance"  # À ajuster selon votre structure de projet
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
        
        # Récupérer la sortie
        output = result.stdout
        
        # Analyser la sortie pour extraire les informations
        question_match = re.search(r"Question : (.*)", output)
        true_answer_match = re.search(r"True Answer : (.*)", output)
        predicted_answer_match = re.search(r"Predicted Answer : (.*)", output)
        meteor_match = re.search(r"(METEOR Score:.*)", output)
        bleu_match = re.search(r"(BLEU Score:.*)", output)
        bert_match = re.search(r"(BERT Score Precision:.*)", output)
        rouge_match = re.search(r"(rougeScores:.*)", output)
        
        # Extraire les valeurs (ou utiliser des chaînes vides si non trouvées)
        question = question_match.group(1).strip() if question_match else ""
        true_answer = true_answer_match.group(1).strip() if true_answer_match else ""
        predicted_answer = predicted_answer_match.group(1).strip() if predicted_answer_match else ""
        meteor = meteor_match.group(1).strip() if meteor_match else ""
        bleu = bleu_match.group(1).strip() if bleu_match else ""
        bert = bert_match.group(1).strip() if bert_match else ""
        rouge = rouge_match.group(1).strip() if rouge_match else ""
        
       
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
        
        # Pour éviter les problèmes d'échappement des caractères spéciaux dans les arguments
        # Encodage sécuritaire pour passer des arguments à subprocess
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
    # venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    # domain = request.args.get('domain')
    # result = subprocess.run(
    #     [venv_python, '-m', f'{domain}.getAllQA'],
    #     capture_output=True,
    #     text=True,
    #     check=True
    #     )

    # response= jsonify({'response': result.stdout})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    # return response
    domain = request.args.get('domain')
    module = importlib.import_module(f'{domain}.getAllQA')
    data = module.getAllQA()
   
    return jsonify(data)


@app.route('/calculateAverages', methods=['GET'])
def calculate_averages():

    domain = request.args.get('domain', 'finance')
    
    # Mock data - simulated average metrics for different nChunks values
    averages = {}
    
    # Generate mock data for nChunks from 0 to 10
    for n_chunks in range(11):
        # Create some realistic looking mock data
        # Using a seed based on nChunks to create a pattern in the results
        base_value = 0.5 + (n_chunks / 20)  # Values increase slightly with more chunks
        
        # Add some randomness but keep within reasonable bounds
        meteor = min(0.95, max(0.3, base_value + random.uniform(-0.05, 0.05)))
        bleu = min(0.85, max(0.2, base_value - 0.1 + random.uniform(-0.05, 0.05)))
        bert = min(0.9, max(0.4, base_value + 0.05 + random.uniform(-0.05, 0.05)))
        rouge = min(0.9, max(0.25, base_value - 0.05 + random.uniform(-0.05, 0.05)))
        

        averages[n_chunks] = {
            "meteor": meteor,
            "bleu": bleu,
            "bert": bert,
            "rouge": rouge
        }
    
    return jsonify(averages)


   


if __name__ == '__main__':
    app.run(debug=True)



