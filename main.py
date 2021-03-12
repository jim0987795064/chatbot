## Import required modules
# Import system modules
import os

# Import 3rd-party modules

# Import self-written modules
#from model import (
 #   chat_bow,chat_tfidf
#)
from model import chat_bow
from model import chat_tfidf
## GLOBAL variables
# TIME = time.time()





## Main script
if __name__ == '__main__':
    x=[input("YOUR QUERY: \n")]
    while(x[0].lower() not in ['bye','thanks','ok','cya']):
        print("\n\nBOT: ",chat_tfidf(x))
        x=[input("YOUR QUERY :")]
    print("\nBye !! Stay Safe!!")

   
