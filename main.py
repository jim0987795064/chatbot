## Import required modules
# Import system modules
import os
import sys


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

def main(argv):
    ## Preprocess corpus
    #corpus, train_data, eval_data = preprocess()
    corpus = preprocess()
    ## Run model (embedding)
    if argv[1] == "tfidf":
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


    main(sys.argv)

    # while(x[0].lower() not in ['bye','thanks','ok','cya']):
    #     print("\n\nBOT: ", chat_tfidf(x))
    #     x = input("YOUR QUERY :")
    # print("\nBye !! Stay Safe!!")

   
