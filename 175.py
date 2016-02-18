import csv
import numpy as np
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import re
import random


if __name__ == "__main__":
    data = []
    with open('175_data.csv', 'rU') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data.append(row)
    data = np.array(data)[1:]
    
    inst = data[:41, 3]
    story = data[:41,4]
    bag = defaultdict(int)
    sw = set(stopwords.words('english'))

    #get a random quest's grammar pattern.
    patterns = defaultdict(int)
    for x in story:
        tokens = nltk.word_tokenize(x)
        tagged_tokens  = nltk.pos_tag(tokens)
        pattern = tuple(i[1] for i in tagged_tokens)
        if pattern in patterns:
            patterns[pattern] += 1
        else:
            patterns[pattern] = 1        
    chosen_pattern = random.choice(list(patterns.keys()))
    print(chosen_pattern)

    #get a bag of tagged words.
    tagged_bag = defaultdict(int)
    for s in story:
        tokens =  nltk.word_tokenize(s)
        tagged_tokens = nltk.pos_tag(tokens)
        for t in tagged_tokens:
            if t in tagged_bag:
                tagged_bag[t] +=1
            else:
                tagged_bag[t] = 1

    #for each tag, get word list.
    grammar_bag = {}
    for t in tagged_bag:
        if t[1] in grammar_bag:
            grammar_bag[t[1]].append(t[0])
        else:
            grammar_bag[t[1]] = [t[0]]


    #generate sentence based on grammar.
    result = ''
    for pos in chosen_pattern:
        result += random.choice(grammar_bag[pos]) + ' '
    print(result)
        
        
        
    
    
    
        
    
    
        
        
    

  
    
    
    
    

        
    
    

            
