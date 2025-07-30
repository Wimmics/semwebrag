import csv
import os
import code



def getAllQA(code="NVDA"):
    with open("callForTenderClassic/allQA.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
        questions = []
        answers = []
        for row in rows:

            questions.append(row["Question"])
            answers.append(row["Reponse"])
        result = {
            "questions": questions,
            "answers": answers
        }

        return result
