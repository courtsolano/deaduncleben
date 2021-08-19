import numpy as np
from collections import Counter


def read_data(text):
    words = text.split()
    corpusLength = len(words)
    freqs = Counter(words)
    unigram_dict = {"<unk>": 0}
    for key in freqs:  # create a new dict with <unk>
        if freqs[key] == 1:
            unigram_dict["<unk>"] +=1
        else:
            unigram_dict[key] = freqs[key]
    return words, unigram_dict, corpusLength


def clean_data(words, unigram_dict):
    words_unk = []  # clean the data by adding <unk>
    for i in range(len(words)):
        if words[i] in unigram_dict.keys():
            words_unk.append(words[i])
        else:
            words_unk.append("<unk>")

    return get_bigrams(add_one_smoothing(words_unk), words_unk)


def uni_prob(word, freqs, length):
    if word in freqs:
        return freqs[word] / length
    else:
        return freqs["<unk>"] / length


def product(nums):
    prod = 1
    for num in nums:
        prod = prod * num
    return prod


def sentence_prob(words, freqs, length):
    return product(uni_prob(word, freqs, length) for word in words)


def get_bigrams(bigrams, words):  # get all the bigrams in the training set
    for i in range(len(words) - 1):
        bigrams[(words[i], words[i + 1])] += 1
    return bigrams


def add_one_smoothing(words):
    bigrams = {}
    words = set(words)
    for i in words:
        for j in words:
            bigrams[(i, j)] = 0.01  # For shannon
    return bigrams


def bi_prob(i, j, bigrams, unigram):
    numerator = bigrams[(i, j)]
    denominator = unigram[i] + (0.01 * len(unigram.keys()))  # for shannon
    return numerator / denominator


def sentence_prob_bi(words, bigrams, unigrams):
    p = 1
    for k in range(len(words) - 1):
        i = words[k]
        j = words[k + 1]
        if words[k] not in unigrams:  # need to check if the word exists.
            i = "<unk>"
        if words[k + 1] not in unigrams:  # if not, make it unk
            j = "<unk>"
        p = p * bi_prob(i, j, bigrams, unigrams)  # multiple all bigram probs together of sentence
    return p


def generate_tweet(unigram, bigram, d):
    sentence = "<s> "
    word = "<s>"
    while word != "</s>":
        for w in unigram.keys():
            n = np.random.uniform(0, 1)  # get a random value
            if n - bigram[(word, w)] < 0:  # once the random value is within the chunk of prob for that bigram pick that
                word = w
                if w == '<unk>':
                    sentence += np.random.choice(d['NN'])
                else:
                    sentence += w
                sentence += " "
            else:
                n = n - bigram[(word, w)]
        if len(sentence) > 220:
            word = "</s>"
            sentence += word
            sentence += " "
    return sentence
