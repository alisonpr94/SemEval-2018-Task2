# -*- coding: utf-8 -*-

#@author: Alison Ribeiro

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

def preprocessing(tweets):
    tweets = tokenize(tweets)
    tweets = remove_stopwords(tweets)
    tweets = remove_url(tweets)
    tweets = my_stem(tweets)
    return tweets

def tokenize(tweets):
    tknzr = TweetTokenizer()
    tokenized_tweets = []
    for tweet in tweets:
        tokenized_tweets.append(tknzr.tokenize(tweet))
    return tokenized_tweets

def remove_stopwords(tweets):
    stopwords =  nltk.corpus.stopwords.words('english')
    stopwords.extend(("im","dont","wont"))
    tweets_nostop = []
    for tweet in tweets:
        tweet_nostop = [w.lower() for w in tweet if w.lower() not in stopwords]
        tweets_nostop.append(tweet_nostop)
    return tweets_nostop

def remove_url(tweets):
    tweets_filtered = []
    regex = re.compile("^https?:\/\/.*[\r\n]*")
    for tweet in tweets:
        tweet_filtered = [w for w in tweet if not regex.match(w)]
        tweets_filtered.append(tweet_filtered)
    return tweets_filtered
 
def my_stem(tweets):
    ps = PorterStemmer()
    tweets_stemmed = []
    for tweet in tweets:
        tweet_stemmed = [ps.stem(tweet)]
        tweets_stemmed.append(tweet_stemmed)
    return tweets_stemmed