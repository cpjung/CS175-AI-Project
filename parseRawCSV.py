import csv
import pandas as pd

quests = pd.read_csv("quests.csv")
#print(list(quests.columns.values))
details = quests.Details
rawStr = ""
for d in details:
    rawStr += str(d)

import nltk

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print word + " "
        word = cfdist[word].max()

tokens = nltk.word_tokenize(rawStr)
text = nltk.Text(tokens)
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

generate_model(cfd, 'King')
