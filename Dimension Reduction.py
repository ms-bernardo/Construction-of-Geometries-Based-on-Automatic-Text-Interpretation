from __future__ import absolute_import, division, print_function
import numpy as np
import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE

# Open the trained model
model = w2v.Word2Vec.load("Model.w2v")
print("Model loaded")

# Vocabulary length
vocab_len = len(model.wv.vocab)
print("Vocabulary length is ", vocab_len)

# Define Matix
word_vectors_matrix = np.ndarray(shape=(vocab_len, 300), 
                                    dtype='float64')
word_list = []
i = 0

# Fill the Matix
for word in model.wv.vocab:
    word_vectors_matrix[i] = model[word]
    word_list.append(word)
    i += 1
    if i == vocab_len:
        break


# Compress the word vectors into 2D space
tsne = TSNE(n_components = 2, random_state = 0, metric="cosine")
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)

# Save Matrix
np.save("Mtx_name", word_vectors_matrix)
np.save("Mtx_2d_name", word_vectors_matrix_2d)
