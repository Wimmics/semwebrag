import json
import csv


#Scripts pour trasformer le début de covidQA en csv (utilisé pour l'évaluation)



# Chemin du fichier JSON (à adapter selon votre environnement)
input_file = '200421_covidQA.json'

def extract_first_paragraph_qa(file_path):
    # Lire le contenu du fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouver la partie JSON dans le fichier
    json_start = content.find('{')
    json_end = content.find('sachant qu')
    json_str = content[json_start:json_end].strip()
    
    # Corriger la chaîne JSON si elle est incomplète
    if not json_str.endswith('}'):
        json_str += '}'
    
    # Charger la chaîne JSON
    try:
        data = json.loads(json_str)
        
        # Vérifier si les données sont structurées comme prévu
        if 'data' in data and len(data['data']) > 0:
            if 'paragraphs' in data['data'][0] and len(data['data'][0]['paragraphs']) > 0:
                # Extraire les questions et réponses du premier paragraphe uniquement
                qa_pairs = []
                for qa in data['data'][0]['paragraphs'][0]['qas']:
                    question = qa['question']
                    # Prendre la première réponse si plusieurs sont disponibles
                    answer = qa['answers'][0]['text'] if qa['answers'] else "Pas de réponse"
                    qa_pairs.append({'question': question, 'answer': answer})
                
                return qa_pairs
    except json.JSONDecodeError as e:
        print(f"Erreur lors du décodage JSON: {e}")
    
    return []

def save_to_csv(qa_pairs, output_file):
    # Enregistrer les paires Q/R dans un fichier CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question', 'Réponse'])
        for pair in qa_pairs:
            writer.writerow([pair['question'], pair['answer']])

# Extraire les Q/R et les enregistrer
qa_pairs = extract_first_paragraph_qa(input_file)
if qa_pairs:
    output_file = 'questions_reponses.csv'
    save_to_csv(qa_pairs, output_file)
    print(f"{len(qa_pairs)} paires question/réponse ont été extraites et enregistrées dans {output_file}")
else:
    print("Aucune paire question/réponse n'a été trouvée dans le premier paragraphe.")