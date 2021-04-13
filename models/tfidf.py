import pandas as pd


from preprocess import text_normalization
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from sklearn.metrics.pairwise import cosine_similarity

from Evaluation_Methods.bleu import bleu

class TFIDFModel():
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def fit(self, corpus):
        """
        Arguments:
            - corpus (pd.DataFrame): The corpus that has `Context`, `Answer`, 
                                     and `Normalized` text

        Returns:
            - None: the vectorizer, embeddings, and vocab of the model is
                    initialized
        """
        # Fit corpus on tf-idf
        print("Fitting TF-IDF...")
        self.corpus = corpus

        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer
        )
        # Q x V
        self.corpus_embedding = self.vectorizer.fit_transform(
            corpus['Normalized'],
        ).toarray()

        self.vocab = self.vectorizer.get_feature_names()


    def chat(self, text):
        user_input = text
        text = text_normalization(text)
        text = self.vectorizer.transform([text]).toarray()
        
        cos = cosine_similarity(
            self.corpus_embedding,
            text, 
        )

        index_value = cos.argmax()
       
        return self.corpus['Answer'].loc[index_value]
    





