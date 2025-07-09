import json
import csv
import sys

# Script modifié pour transformer un nombre défini de paragraphes de covidQA en fichier CSV
input_file = '200421_covidQA.json'

def extract_paragraphs_qa(file_path, num_paragraphs=1):

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    json_start = content.find('{')
    json_str = content[json_start:].strip()
    
    try:
        data = json.loads(json_str)
        
        if 'data' in data and len(data['data']) > 0:
            all_paragraphs = []
            for data_item in data['data']:
                if 'paragraphs' in data_item:
                    all_paragraphs.extend(data_item['paragraphs'])
            
            if all_paragraphs and len(all_paragraphs) > 0:
                max_paragraphs = min(num_paragraphs, len(all_paragraphs))
                
                print(f"Nombre total de paragraphes disponibles: {len(all_paragraphs)}")
                print(f"Extraction de {max_paragraphs} paragraphe(s)...")
                
                qa_pairs = []
                for i in range(max_paragraphs):
                    paragraph = all_paragraphs[i]
                    print(f"Extraction du paragraphe {i+1}/{max_paragraphs}...")
                    
                    for qa in paragraph['qas']:
                        question = qa['question']
                        answer = qa['answers'][0]['text'] if qa['answers'] else "Pas de réponse"
                        
                        qa_pairs.append({
                            'question': question, 
                            'answer': answer,
                            'paragraph_id': i+1
                        })
                
                print(f"Total: {len(qa_pairs)} paires question/réponse extraites de {max_paragraphs} paragraphe(s)")
                return qa_pairs
                
    except json.JSONDecodeError as e:
        print(f"Erreur lors du décodage JSON: {e}")
    
    return []

def save_to_csv(qa_pairs, output_file):

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question', 'Réponse', 'Paragraphe_Source'])
        for pair in qa_pairs:
            writer.writerow([pair['question'], pair['answer'], pair['paragraph_id']])

def main():

    num_paragraphs = 1
    
    if len(sys.argv) > 1:
        try:
            num_paragraphs = int(sys.argv[1])
            if num_paragraphs <= 0:
                print("Le nombre de paragraphes doit être un entier positif.")
                return
        except ValueError:
            print("Veuillez fournir un nombre entier valide pour le nombre de paragraphes.")
            return
    
    print(f"Extraction de {num_paragraphs} paragraphe(s)...")
    
    qa_pairs = extract_paragraphs_qa(input_file, num_paragraphs)
    
    if qa_pairs:
        output_file = f'questions_reponses_{num_paragraphs}paragraphes.csv'
        save_to_csv(qa_pairs, output_file)
        print(f"{len(qa_pairs)} paires question/réponse ont été extraites et enregistrées dans {output_file}")
    else:
        print("Aucune paire question/réponse n'a été trouvée.")

def extract_n_paragraphs(n_paragraphs):

    qa_pairs = extract_paragraphs_qa(input_file, n_paragraphs)
    
    if qa_pairs:
        output_file = f'questions_reponses_{n_paragraphs}paragraphes.csv'
        save_to_csv(qa_pairs, output_file)
        print(f"{len(qa_pairs)} paires question/réponse ont été extraites et enregistrées dans {output_file}")
    else:
        print("Aucune paire question/réponse n'a été trouvée.")

if __name__ == "__main__":
    main()
