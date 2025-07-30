import argparse

# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
import requests




# def chat_completion(question: str):
#     with open("key.txt", "r", encoding="utf-8") as file:
#         key = file.read().strip()

#     headers = {
#         "Authorization": f"Bearer {key}",
#         "Content-Type": "application/json"
#     }
 
#     url = "https://api.groq.com/openai/v1/chat/completions"
    
#     payload = {
#         "model": "llama3-70b-8192",  
#         "messages": [
#             {"role": "system", "content": "You are Nestor, a virtual assistant. Answer to the question by using the context given bellow."},
#             {"role": "user", "content": question}
#         ],
#         "max_tokens": 1500
#     }
    
#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         result = response.json()
#         content = result["choices"][0]["message"]["content"]
#         print(f" {content}")
#         return content
#     else:
#         error_message = f"Erreur: {response.status_code} - {response.text}"
#         print(error_message)
#         return error_message


def chat_completion(question: str):
    with open("keyForgeron.txt", "r", encoding="utf-8") as file:
        key = file.read().strip().replace('\n', '').replace('\r', '').replace(' ', '')
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # URL de l'API Mistral
    url = "https://api.mistral.ai/v1/chat/completions"
    
    # Payload pour la requête
    payload = {
        "model": "mistral-medium-latest",  # ou "mistral-medium-latest", "mistral-small-latest"
        "messages": [
            {"role": "system", "content": "You are Nestor, a virtual assistant. Answer to the question by using the context given bellow."},
            {"role": "user", "content": question}
        ],
        "max_tokens": 1500
    }
    
    try:
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
            
    except requests.exceptions.RequestException as e:
        error_message = f"Erreur de connexion: {str(e)}"
        print(error_message)
        return error_message

if __name__ == '__main__' : 
    text= ''
    with open ("medical/query_enrichie.txt", "r", encoding="utf-8") as file:
        text = file.read()

    chat_completion(text)

