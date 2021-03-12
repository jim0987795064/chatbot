
# Import system modules
import os
import json
import re
import math

# Import 3rd-party modules
import pandas as pd
import numpy as np
import nltk 



from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words

# Import self-written modules




## Function

def preprocess():
    # Load data and files store all jsonl names
    files=[]
    contents=os.listdir('/mnt/d/cs/CsieProject/chatbot/data/raw')
    for i in contents:
        if i.endswith('.jsonl'):
            files.append(i)

    # Open all jsonl files ,and File stores all jsonl files

    File=list()
    for i in files:
        i='/mnt/d/cs/CsieProject/chatbot/data/raw/'+i
        with open(i,'r') as data:
            data=list(data)
            File.extend(data)

    question=[]
    answer=[]

    # Take all json files in File out,and store questionText and answerText in lists 
    for i in range(8498):
        data=json.loads(File[i])
        question.append(data["questionText"])
        answer.append(data["answerText"])

    # Let question list and answer list store in a dict named data

    data={}
    data["Context"]=question
    data["Answer"]=answer

    return data










# function that performs text normalization steps
def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'                                                        
        else:
            pos_val='n' # Noun
        lema_token=lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence




# Function that removes stop words and process the text
def tokenize_stopword_lemmatize(text):   
    tag_list = pos_tag(nltk.word_tokenize(text),tagset=None)
    stop = stopwords.words('english')
    lema = wordnet.WordNetLemmatizer()
    lema_word = []
    for token,pos_token in tag_list:
        if token in stop:
            continue
        
        if pos_token.startswith('V'):
            pos_val = 'v'
        elif pos_token.startswith('J'):
            pos_val = 'a'
        elif pos_token.startswith('R'):
            pos_val = 'r'
        else:
            pos_val = 'n'
        
        lema_token = lema.lemmatize(token,pos_val)
        lema_word.append(lema_token)
    
    return " ".join(lema_word)

# Make dict "data" to DataFrame
df=pd.DataFrame(preprocess())


df['lemmatized_text']=df['Context'].apply(text_normalization)

cv = CountVectorizer() 
tfidf=TfidfVectorizer() # intializing tf-id 

# Count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()
features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)

# tf_idf
x_tfidf=tfidf.fit_transform(df['lemmatized_text']).toarray() # transforming the data into array
df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names()) 





if __name__ == "__main__":
    # Test for this file
    print("no hello")