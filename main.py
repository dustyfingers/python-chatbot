import nltk 
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow
import random
import json

stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokenize words in pattern
        wrds = nltk.word_tokenize(pattern)
        # put wrds into words array from above
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])


# stem words, remove dupes etc
words = [stemmer.stem(w.lower()) for w in words]
# set() removes dupes, list() converts back to list, sorted() sorts the items
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for wrd in wrds:
        if wrd in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)