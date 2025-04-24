import argparse

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


def chat_completion(question: str):
  key = ''
  with open("key.txt", "r", encoding="utf-8") as file:
      key = file.read()
  # no need to use a token
  model = ChatMistralAI(model="Meta-Llama-3_1-70B-Instruct", 
                        api_key=key,
                        endpoint='https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1', 
                        max_tokens=1500)


  prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Nestor, a virtual assistant. Answer to the question by using the context given bellow."),
    ("human", "{question}"),
  ])

  chain = prompt | model

  response = chain.invoke(question)

  print(f" {response.content}")
  return response.content


if __name__ == '__main__' : 
    text= ''
    with open ("finance/query_enrichie.txt", "r", encoding="utf-8") as file:
        text = file.read()

    chat_completion(text)