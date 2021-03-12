# Import system modules
import os
import json
import re
import math

# Import 3rd-party modules
import pandas as pd
import numpy as np
import nltk 
import nltk 
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop word

# Import self-written modules
from preprocess import text_normalization
from preprocess import tokenize_stopword_lemmatize

from preprocess import (
    df_tfidf,
    df,
    tfidf,
    cv
)



# defining a function that returns response to query using bow
def chat_bow(text):
    s = tokenize_stopword_lemmatize(text)
     # calling the function to perform text normalization
    lemma = text_normalization(s)
    bow = cv.transform([lemma]).toarray() # applying bow
    cosine_value = 1- pairwise_distances(df_bow, bow, metric='cosine' )
    index_value = cosine_value.argmax() # getting index value 
    return df['Answer'].loc[index_value]

# defining a function that returns response to query using tf-idf
def chat_tfidf(text):
    # calling the function to perform text normalization
    lemma = text_normalization(text) 
    tf = tfidf.transform([lemma]).toarray() # applying tf-idf
    df_tf = pd.DataFrame(tf, columns=tfidf.get_feature_names())
    # applying cosine similarity
    cos = 1-pairwise_distances(df_tfidf, df_tf, metric='cosine') 
    index_value = cos.argmax() # getting index value
    return df['Answer'].loc[index_value]

    