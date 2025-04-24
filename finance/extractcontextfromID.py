#but du script : charger le fichier Financial-QA-10k.csv
#pour chaque ligne, on regarde la colone "ticker"
#si la valeur de cette colonne correspond a celle passée en paramètre, on extrait le contenu de la colonne "context"
# les différent contexte seront stockés à la suite dans un fichier texte nommé financialText.txt

import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python extractcontextfromID.py <ticker>")
    sys.exit(1)

ticker = sys.argv[1]
print("Extracting context for ticker", ticker)

with open("Financial-QA-10k.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    with open("financialText.txt", "w", encoding="utf-8") as out:
        for row in reader:
            if row["ticker"] == ticker:
                out.write(row["context"])
                out.write("\n\n")
                print("Context extracted for", ticker)
                
        else:
            print("Ticker not found")

print("Context extraction complete")

#fin du script
