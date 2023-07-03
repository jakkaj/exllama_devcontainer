import numpy as np
from sentence_transformers import SentenceTransformer
sentences = ["The boy was warm", "The girl was cold", "the boy is warm", "there was a beach"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

embeddings1 = embeddings[0]
embeddings2 = embeddings[1]
embeddings3 = embeddings[2]
embeddings4 = embeddings[3]

dot_product1 = np.dot(embeddings1, embeddings2)
dot_product2 = np.dot(embeddings1, embeddings3)
dot_product3 = np.dot(embeddings1, embeddings4)


print(dot_product1)
print(dot_product2)
print(dot_product3)