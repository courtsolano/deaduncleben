import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import TweetTokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def tokenize(clean_text):
    d = {}
    pos_list = []
    tokenizer = TweetTokenizer()

    sentences = []
    sentence = ""
    for w in str.split(clean_text):
        if w != "<s>":
            if w != "</s>":
                sentence += w + " "
            else:
                sentence = sentence[:-1]
                sentences.append(sentence)
                sentence = ""

    for s in sentences:
        t = tokenizer.tokenize(s)
        a = nltk.pos_tag(t)
        pos_list.append(a)
        for v in a:
            if v[1] in d:
                d[v[1]].append(v[0])
            else:
                d[v[1]] = [v[0]]
    return pos_list, d


def generate_tweet(pos_list, d):
    # so now what we do is every day we pick a new np.random.choice(poslist),
    # look at the parts of speech associated, and pick a random word from the dictionary of that part of speech
    tweet = ''
    punctuation_list = [".", ",", "?", "!", "'"]
    temp = np.random.choice(pos_list)
    for words in temp:
        new_word = np.random.choice(d[words[1]])
        if new_word[0] not in punctuation_list:
            tweet += ' '
        tweet += new_word
    tweet = tweet[1:]
    return tweet
