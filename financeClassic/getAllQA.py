import csv
import os
import code



def getAllQA(code="NVDA"):
    with open("finance/Financial-QA-10k.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row["ticker"] == code]
        questions = []
        answers = []
        for row in rows:

            questions.append(row["question"])
            answers.append(row["answer"])
        result = {
            "questions": questions,
            "answers": answers
        }

        return result
