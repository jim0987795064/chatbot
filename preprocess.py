
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

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
# Import self-written modules

## Function
def preprocess():
    print("Loading files")
    # Load data and files store all jsonl names
    files = list()
    ###Do not write dead and shorten the code
    contents = os.listdir('/mnt/d/cs/CsieProject/chatbot/data/raw')
    for i in contents:
        if i.endswith('.jsonl'):
            files.append(i)

    # Open all jsonl files ,and File stores all jsonl files

    File = list()
    for i in files:
        i = '/mnt/d/cs/CsieProject/chatbot/data/raw/'+i
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
    """
    #This part will process evaluation data
    train_data_len = round(len(File)*0.8)
    total_data_len = len(File)
    question.clear()
    answer.clear()

    for i in trange (train_data_len):
        data = json.loads(File[i])
        question.append(data["questionText"])
        answer.append(data["answerText"])

    train_data = {}
    train_data["Context"] = question
    train_data["Answer"] = answer
    df_train = pd.DataFrame(train_data)
    df_train['Normalized'] = df_train['Context'].apply(text_normalization)
    

    question.clear()
    answer.clear()
    
    for i in trange (train_data_len, total_data_len):
        data = json.loads(File[i])
        question.append(data["questionText"])
        answer.append(data["answerText"])
    
    eval_data = {}
    eval_data["Context"] = question
    eval_data["Answer"] = answer
    df_eval = pd.DataFrame(eval_data)
    df_eval['Normalized'] = df_eval['Context'].apply(text_normalization)"""

    return df#, df_train, df_eval


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

# Function that removes stop words and process the text
#def tokenize_stopword_lemmatize(text):   
#   tag_list = pos_tag(
#      nltk.word_tokenize(text),
#      tagset=None
#   )
#   stop = stopwords.words('english')
#   lema = wordnet.WordNetLemmatizer()
#   lema_word = []
#   for token,pos_token in tag_list:
#     if token in stop:
#           continue
#   
#   if pos_token.startswith('V'):
#       pos_val = 'v'
#   elif pos_token.startswith('J'):
#       pos_val = 'a'
#   elif pos_token.startswith('R'):
#       pos_val = 'r'
#   else:
#       pos_val = 'n'
#   
#   lema_token = lema.lemmatize(token,pos_val)
#   lema_word.append(lema_token)
    
 #   return " ".join(lema_word)

# Make dict "data" to DataFrame
#corpus = preprocess()
#df = pd.DataFrame(corpus) #store in csv

#df['lemmatized_text'] = df['Context'].apply(text_normalization)


#tfidf = TfidfVectorizer() # intializing tf-id 

## tf_idf
#x_tfidf = tfidf.fit_transform(df['lemmatized_text']).toarray() # transforming the data into array
#df_tfidf = pd.DataFrame(x_tfidf,columns = tfidf.get_feature_names()) 

if __name__ == "__main__":
    # Test for this file
    print("no hello")