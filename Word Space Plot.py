from __future__ import absolute_import, division, print_function
import numpy as np
import gensim.models.word2vec as w2v
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Open the trained model
model = w2v.Word2Vec.load("Model.w2v")
print("Model loaded")

# Vocabulary length
vocab_len = len(model.wv.vocab)
print("Vocabulary length is ", vocab_len)

# Open the trained model matrix
word_vectors_matrix_2d = np.load("Mtx_name.npy")

# Open the multi-dimensional matrix
word_vectors_matrix = np.load("Mtx_2d_name.npy")

word_list = []
i = 0
for word in model.wv.vocab:
    word_list.append(word)
    i += 1
    if i == vocab_len:
        break

# Word points DataFrame
points = pd.DataFrame([
    (word, coords[0], coords[1])
    for word, coords in [
        (word, word_vectors_matrix_2d[word_list.index(word)])
        for word in word_list
    ]
], columns=["Word", "x", "y"])

sns.set_context("poster")
fig, ax = plt.subplots()

# Plot the word points
ax.plot(points.x, points.y, 'ro', markersize=15)

# Defining Axes
offset = 1.0
ax.set_xlim(min(points.x) - offset, max(points.x) + offset)
ax.set_ylim(min(points.y) - offset, max(points.y) + offset)

# Plot the point tags
k = 0
for i, j in zip(points.x, points.y):
    corr = -0.05  # correction for annotation in marker
    ax.annotate(points.Word.values[k], xy=(i + corr, j + corr))
    k += 1

plt.show()
