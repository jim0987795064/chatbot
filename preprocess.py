
# Import system modules
import os
import json
import re
import math

# Import 3rd-party modules
import pandas as pd
import numpy as np
import nltk 

from tqdm import trange


from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words
import nltk

# Import self-written modules

## Function
def preprocess():
    print("Loading files")
    # Load data and files store all jsonl names
    files = list()
    ###Do not write dead and shorten the code
    path = os.getcwd()+'/data/raw/'
    contents = os.listdir(path)
    for i in contents:
        if i.endswith('.jsonl'):
            files.append(i)

    # Open all jsonl files ,and File stores all jsonl files

    File = list()
    for i in files:
        i = path+i
        with open(i,'r') as data:
            data = list(data)
            File.extend(data)

    question = list()
    answer = list()

    # Take all json files in File out,and store questionText and answerText in lists 
    for i in trange(len(File)):
        data = json.loads(File[i])
        question.append(data["questionText"])
        answer.append(data["answerText"])

    # Let question list and answer list store in a dict named data

    data = {}
    data["Context"] = question
    data["Answer"] = answer

    df = pd.DataFrame(data)
    print("Normalize questions")
    df['Normalized'] = df['Context'].apply(text_normalization)

    df.to_csv("preprocessed.csv")
    
    return df


# function that performs text normalization steps
def text_normalization(text):
    text = str(text).lower() # text to lower case
    spl_char_text = re.sub(r'[^ a-z]','',text) # removing special characters
    tokens = nltk.word_tokenize(spl_char_text) # word tokenizing
    lema = wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list = pos_tag(tokens,tagset = None) # parts of speech
    lema_words = []   # empty list 
    
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'): # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'): # Adverb
            pos_val = 'r'                                                        
        else:
            pos_val = 'n' # Noun
        lema_token = lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence


if __name__ == "__main__":
    # Test for this file
    pass