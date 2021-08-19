from twython import Twython
from auth import (
    consumer_key,
    consumer_secret,
    access_token,
    access_token_secret
)
import numpy
import shannon
import perceptron


def main():
    twitter = Twython(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )

    clean_text = read_data("tweetdata.txt")

    # part 1
    pos_list, d = perceptron.tokenize(clean_text)

    # part 2
    words, unigram_dict, corpus_length = shannon.read_data(clean_text)
    bigram_dict = shannon.clean_data(words, unigram_dict)

    bigram_prob_dict = {}
    for key in bigram_dict:
        bigram_prob_dict[key] = shannon.bi_prob(key[0], key[1], bigram_dict, unigram_dict)

    # choose a tweet
    tweets = [perceptron.generate_tweet(pos_list, d), shannon.generate_tweet(unigram_dict, bigram_prob_dict, d)[4:-6]]
    print(tweets)
    message = numpy.random.choice(tweets)

    twitter.update_status(status=message)
    print("Tweeted: {}".format(message))


def read_data(text):
    return open(text).read()


if __name__=="__main__":
    main()