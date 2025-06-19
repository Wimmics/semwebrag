import json
import csv



#Script made to transform the beginning of covidQA into a csv file (used for evaluation)


input_file = '200421_covidQA.json'

def extract_first_paragraph_qa(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    
    json_start = content.find('{')
    json_end = content.find('sachant qu')
    json_str = content[json_start:json_end].strip()
    
    if not json_str.endswith('}'):
        json_str += '}'
    
    try:
        data = json.loads(json_str)
        
        # Check if 'data' and 'paragraphs' exist and are not empty
        if 'data' in data and len(data['data']) > 0:
            if 'paragraphs' in data['data'][0] and len(data['data'][0]['paragraphs']) > 0:
                #only extract questions and answers from the first paragraph
                qa_pairs = []
                for qa in data['data'][0]['paragraphs'][0]['qas']:
                    question = qa['question']
                    #Take the first answer if it exists, otherwise set to "Pas de réponse"
                    answer = qa['answers'][0]['text'] if qa['answers'] else "Pas de réponse"
                    qa_pairs.append({'question': question, 'answer': answer})
                
                return qa_pairs
    except json.JSONDecodeError as e:
        print(f"Erreur lors du décodage JSON: {e}")
    
    return []

def save_to_csv(qa_pairs, output_file):
    # save Q/A in a csv file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question', 'Réponse'])
        for pair in qa_pairs:
            writer.writerow([pair['question'], pair['answer']])

qa_pairs = extract_first_paragraph_qa(input_file)
if qa_pairs:
    output_file = 'questions_reponses.csv'
    save_to_csv(qa_pairs, output_file)
    print(f"{len(qa_pairs)} paires question/réponse ont été extraites et enregistrées dans {output_file}")
else:
    print("Aucune paire question/réponse n'a été trouvée dans le premier paragraphe.")