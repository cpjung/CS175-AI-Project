import csv
import re
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
        #print(word + " ")
        print word + " "
        word = cfdist[word].max()
        
def format_simple(document):
    doc = document.lower()                                           #convert to lower case
    doc = re.sub(r'[\.\?\{\}\[\]\\\|\(\)!,:\'-;\"]','',doc)         #remove punctuations
    doc = re.sub(r'[\s]+',' ', doc)                                  #remove whitespace clumps
    doc =  doc.strip()                                               #remove trailing whitespace
    return doc

rawStr = format_simple(rawStr)
tokens = nltk.word_tokenize(rawStr)
text = nltk.Text(tokens)
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)

generate_model(cfd, 'king')
