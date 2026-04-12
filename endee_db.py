import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EndeeVectorDB:
    def __init__(self):
        self.embeddings = []
        self.texts = []
        self.sources = []  #track file name

    def add(self, text, embedding, source):
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.sources.append(source)

    def search(self, query_embedding, k=3):
        if not self.embeddings:
            return []

        #Convert to numpy arrays
        embeddings_array = np.array(self.embeddings)

        #Compute similarity
        similarities = cosine_similarity(
            [query_embedding],
            embeddings_array
        )[0]

       
        top_indices = similarities.argsort()[-k:][::-1]

        results = []
        for i in top_indices:
            results.append({
                "text": self.texts[i],
                "source": self.sources[i]
            })

        return results



db = EndeeVectorDB()