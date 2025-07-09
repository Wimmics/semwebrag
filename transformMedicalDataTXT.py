import json
import sys

#script to extract contexts of covidQA
input_file = '200421_covidQA.json'

def extract_contexts(file_path, num_contexts=1):

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
                max_contexts = min(num_contexts, len(all_paragraphs))
                
                print(f"Nombre total de contextes disponibles: {len(all_paragraphs)}")
                print(f"Extraction de {max_contexts} contexte(s)...")
                
                contexts = []
                for i in range(max_contexts):
                    paragraph = all_paragraphs[i]
                    print(f"Extraction du contexte {i+1}/{max_contexts}...")
                    
                    context_text = paragraph.get('context', 'Pas de contexte')
                    document_id = paragraph.get('document_id', 'ID inconnu')
                    
                    num_questions = len(paragraph.get('qas', []))
                    
                    contexts.append({
                        'context_id': i+1,
                        'document_id': document_id,
                        'context_text': context_text,
                        'num_questions': num_questions,
                        'text_length': len(context_text)
                    })
                
                print(f"Total: {len(contexts)} contextes extraits")
                return contexts
                
    except json.JSONDecodeError as e:
        print(f"Erreur lors du décodage JSON: {e}")
    
    return []

def save_contexts_to_txt(contexts, output_file):

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, context in enumerate(contexts):
            f.write(context['context_text'])
            #add blank between contexts
            if i < len(contexts) - 1:
                f.write('\n\n')

def save_contexts_to_json(contexts, output_file):

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)

def display_context_summary(contexts):

    print("\n" + "="*50)
    print("RÉSUMÉ DES CONTEXTES EXTRAITS")
    print("="*50)
    
    total_chars = sum(ctx['text_length'] for ctx in contexts)
    total_questions = sum(ctx['num_questions'] for ctx in contexts)
    
    print(f"Nombre de contextes: {len(contexts)}")
    print(f"Total de caractères: {total_chars:,}")
    print(f"Total de questions associées: {total_questions}")
    print(f"Longueur moyenne par contexte: {total_chars//len(contexts):,} caractères")
    
    print("\nDétail par contexte:")
    for ctx in contexts:
        print(f"  Contexte #{ctx['context_id']}: {ctx['text_length']:,} chars, {ctx['num_questions']} questions")

def main():

    num_contexts = 1
    
    if len(sys.argv) > 1:
        try:
            num_contexts = int(sys.argv[1])
            if num_contexts <= 0:
                print("Le nombre de contextes doit être un entier positif.")
                return
        except ValueError:
            print("Veuillez fournir un nombre entier valide pour le nombre de contextes.")
            return
    
    print(f"Extraction de {num_contexts} contexte(s)...")
    
    contexts = extract_contexts(input_file, num_contexts)
    
    if contexts:
        txt_output = f'contextes_{num_contexts}paragraphes.txt'
        json_output = f'contextes_{num_contexts}paragraphes.json'
        
        save_contexts_to_txt(contexts, txt_output)
        save_contexts_to_json(contexts, json_output)
        
        display_context_summary(contexts)
        
        print(f"\n{len(contexts)} contextes ont été extraits et sauvegardés:")
        print(f"Format texte: {txt_output}")
        print(f"Format JSON: {json_output}")
    else:
        print("Aucun contexte n'a été trouvé.")

def extract_n_contexts(n_contexts):

    contexts = extract_contexts(input_file, n_contexts)
    
    if contexts:
        txt_output = f'contextes_{n_contexts}paragraphes.txt'
        json_output = f'contextes_{n_contexts}paragraphes.json'
        
        save_contexts_to_txt(contexts, txt_output)
        save_contexts_to_json(contexts, json_output)
        display_context_summary(contexts)
        
        print(f"\n {len(contexts)} contextes ont été extraits et sauvegardés:")
        print(f"Format texte: {txt_output}")
        print(f"Format JSON: {json_output}")
    else:
        print("Aucun contexte n'a été trouvé.")

if __name__ == "__main__":
    main()

