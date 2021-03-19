## Import required modules
# Import system modules
import os

# Import 3rd-party modules

# Import self-written modules
#from model import (
 #   chat_bow,chat_tfidf
#)

#from model import chat_tfidf
from models.tfidf import TFIDFModel
from preprocess import preprocess
from preprocess import text_normalization

## GLOBAL variables
# TIME = time.time()

def main(args):
    ## Preprocess corpus
    corpus = preprocess()

    ## Run model (embedding)
    if args["model"] == "tfidf":
        # fit TF-IDF
        model = TFIDFModel()
        pass
    else:
        pass

    # train() / fit()
    model.fit(corpus)
    # eval()

    ## Deploy
    # model.chat() 
    while True:
        text = input("Please Enter :")
        output = model.chat(text)
        print("\n\nAnswer:", output)

## Main script
if __name__ == '__main__':
    args = {
        "model": "tfidf"
    }

    main(args)
    # while(x[0].lower() not in ['bye','thanks','ok','cya']):
    #     print("\n\nBOT: ", chat_tfidf(x))
    #     x = input("YOUR QUERY :")
    # print("\nBye !! Stay Safe!!")

   
