## Import required modules
# Import system modules
import os
import sys
import argparse


# Import 3rd-party modules

# Import self-written modules
#from model import (
 #   chat_bow,chat_tfidf
#)

#from model import chat_tfidf
from models.tfidf import TFIDFModel
from models.bert import BERTModel
from preprocess import preprocess
from preprocess import text_normalization

## GLOBAL variables
# TIME = time.time()

def main(model_name):
    ## Preprocess corpus
    #corpus, train_data, eval_data = preprocess()
    corpus = preprocess()
    ## Run model (embedding)
    if model_name == "tfidf":
        # fit TF-IDF
        model = TFIDFModel()
        pass
    else:
        model = BERTModel()
        pass

    # train() / fit()
    model.fit(corpus)
    #print(train_data)
    #print(eval_data)

    # eval()

    ## Deploy
    # model.chat() 
    while True:
        text = input("Please Enter :")
        output = model.chat(text)
        print("\n\nAnswer:", output)

## Main script
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help = 'currently used model name',
        required = True,
        type = str
    )

    args = parser.parse_args()

    main(args.model)



   
