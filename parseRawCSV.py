import csv
import pandas as pd

quests = pd.read_csv("quests.csv")
#print(list(quests.columns.values))
details = quests.Details
rawStr = ""
for d in details:
    rawStr += str(d)

import nltk
nltk.download('punkt')
tokens = nltk.word_tokenize(rawStr)
text = nltk.Text(tokens)
print(text.generate(3))
