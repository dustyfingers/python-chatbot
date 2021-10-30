import nltk 
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json

stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokenize words in pattern
        wrds = nltk.word_tokenize(pattern)
        # put wrds into words array from above
        words.extend(wrds)
        docs.append(pattern)

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
