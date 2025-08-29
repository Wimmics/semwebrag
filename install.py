import subprocess
import sys


def run(cmd):
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result

run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import spacy


models = ["fr_core_news_md", "en_core_web_md"]

for model in models:
    spacy.cli.download(model)
