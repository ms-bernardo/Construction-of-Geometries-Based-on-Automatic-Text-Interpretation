from __future__ import absolute_import, division, print_function
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

import multiprocessing
import os
import re
import gensim.models.word2vec as w2v

# Open Book
file = open('CLEAN_TEXT.txt', encoding='utf-8', errors='ignore')
book = file.read()
file.close()
print("Book loaded!")

raw_sentences = sent_tokenize(book)

def word_tokenizer(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = word_tokenize(clean)
    return words

sentences = []

for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(word_tokenizer(raw_sentence))

# Tokens Counter
token_count = sum([len(sentence) for sentence in sentences])
print("This corpus contains {} tokens.".format(token_count))

# Train The Model
Model = w2v.Word2Vec(
    sg = 1, #Skip-Gram
    workers = multiprocessing.cpu_count(),
    size = 300,
    min_count = 8,
    window = 8,
    sample = 1e-4
)

Model.build_vocab(sentences)

Model.train(sentences, total_examples=Model.corpus_count,
                       epochs=Model.iter)

# Save The Model
if not os.path.exists("Trained"):
    os.makedirs("Trained")

Model.save(os.path.join("Trained", "Model.w2v"))

