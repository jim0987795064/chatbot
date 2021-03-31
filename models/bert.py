from simpletransformers.language_representation import RepresentationModel

from sklearn.metrics.pairwise import cosine_similarity

class BERTModel():
    def __init__(self):
        pass
    
    def fit(self, corpus):
        print("Fitting BERT...")

        self.corpus = corpus

        self.model = RepresentationModel(
            model_type="bert",
            model_name="bert-base-uncased",
            use_cuda = True
        )
        self.corpus_vectors = self.model.encode_sentences(
            self.corpus['Normalized'], 
            combine_strategy = "mean"
        )

            
    def chat(self, text):
        text_vector = self.model.encode_sentences(
            text, 
            combine_strategy = "mean"
        )

        cos = cosine_similarity(
            self.corpus_vectors,
            text_vector, 
        )
        index_value = cos.argmax()

        return self.corpus['Answer'].loc[index_value]

    