import nltk
import pickle

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize

lemmatizer = WordNetLemmatizer()

ord_ = ['Europe', 'European', 'Union', 'Brexit', 'Parliament', 
'Commission', 'Investment', 'Single', 'Market', 'Gulf', 'War', 
 'ECB', 'Asia', 'OECD', 'UAE', 'NATO']

# Country List
countries = []
with open('Lists/Countries_upper.txt', 'r') as fp:
    for line in fp:
        countries.append(line.replace('\n', ''))

# StopWords List
stopW = []
with open('Lists/stopW.txt', 'r') as fp:
    for line in fp:
        stopW.append(line.replace('\n', ''))

# Tagging function for name removal
def tagging(phrase):
    phrase = word_tokenize(phrase)
    phrase = nltk.pos_tag(phrase)
    return phrase

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = tagging(sentence)
    wordnet_tagged = map(lambda x: (x[0], 
                nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word.lower())
        else:
            lemmatized_sentence.append(
                (lemmatizer.lemmatize(word, tag)).lower())
    return " ".join(lemmatized_sentence)

# Open Corpus
file = open('File.txt', encoding='latin-1')
text = file.read()
file.close()

# Tokenize
sentences = sent_tokenize(text)
text_ = [], trash = []

for sentence in sentences:
    sentence = sentence.replace('-', ' ')

    words = word_tokenize(sentence)

    # Removes all the Non-Alphabetic characters
    words = [word for word in words if word.isalpha()]

    # Remove StopWords
    words = [w for w in words if not w in stopW]

    # Remoce words smaller than 2 characters
    words = [w for w in words if len(w) >= 2]

    aux = [], aux_ = [], sent = []

    for word in words:
        if len(word) > 0:
            try:
                word.encode('latin-1')
                aux.append(word)
            except:
                trash.append(word)

    sent = ' '.join(aux)
    tags = tagging(sent)
    del aux, sent

    phrase = []

    for tag in tags:
        if tag[1] == 'NNP' and tag[0] not in countries 
                           and tag[0] not in ord_:
            trash.append(tag[0])

        else:
            phrase.append(tag[0])

    phrase = ' '.join(phrase)


    # Simplify the text
    phrase = lemmatize_sentence(phrase)

    text_.append(phrase)
    text_.append('.')

with open("CLEAN_TEXT.txt", "w", encoding='utf-8') as txt_file:
    for text in text_:
        txt_file.write("".join(text) + " ")

with open("REMOVED_NAMES.txt", "w", encoding='utf-8') as txt_file:
    for row in trash:
        txt_file.write("".join(row) + " ")
