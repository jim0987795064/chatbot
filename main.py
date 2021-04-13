## Import required modules
# Import system modules
import os
import sys
import argparse


# Import 3rd-party modules

# Import self-written modules



from models.tfidf import TFIDFModel
from models.bert import BERTModel
from preprocess import preprocess
from preprocess import text_normalization


## GLOBAL variables


def main(args):
    ## Preprocess corpus
    corpus = preprocess()
    ## Run model (embedding)
    if args.model == "tfidf":
        # fit TF-IDF
        model = TFIDFModel()
        pass
    else:
        model = BERTModel()
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help = 'currently used model name',
        required = True,
        type = str
    )

    args = parser.parse_args()

    main(args)





   
