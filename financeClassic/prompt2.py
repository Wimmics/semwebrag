import argparse
# from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import requests
import json
import time
# import ollama

#=============OVH===================

# def chat_completion(question: str):
#   key = ''
#   with open("key.txt", "r", encoding="utf-8") as file:
#       key = file.read()
#   # no need to use a token
#   model = ChatMistralAI(model="Meta-Llama-3_1-70B-Instruct", 
#                         api_key=key,
#                         endpoint='https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1', 
#                         max_tokens=1500)


#   prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are Nestor, a virtual assistant. Answer to the question by using the context given bellow."),
#     ("human", "{question}"),
#   ])

#   chain = prompt | model

#   response = chain.invoke(question)

#   print(f" {response.content}")
#   return response.content


#=============Groq===================
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

#=============ollama===================
# def chat_completion(question: str):
#     url = "http://localhost:11434/api/chat"

#     payload = {
#         "model": "llama3:70b",  
#         "messages": [
#             {"role": "system", "content": "You are Nestor, a virtual assistant. Answer the question using the context given below."},
#             {"role": "user", "content": question}
#         ],
#         "stream": False  
#     }

#     response = requests.post(url, json=payload)

#     if response.status_code == 200:
#         result = response.json()
#         content = result["message"]["content"]
#         print(f"{content}")
#         return content
#     else:
#         error_message = f"Erreur: {response.status_code} - {response.text}"
#         print(error_message)
#         return error_message

if __name__ == '__main__' : 
    text= ''
    with open ("financeClassic/query_enrichie.txt", "r", encoding="utf-8") as file:
        text = file.read()

    chat_completion(text)