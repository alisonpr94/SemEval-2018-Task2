# -*- coding: utf-8 -*-

#@author: Alison Ribeiro

import pandas as pd
import numpy as np
import scorer_semeval18
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.tokenize import word_tokenize
import preprocessor as p
from nltk.corpus import stopwords
import string
import re
from nltk.stem.snowball import EnglishStemmer
import pickle
from gensim.models import Word2Vec
from Evaluate import evaluate

all_classes = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, model):
        self.model = model
        self.modelweight = None
       
        self.dim = len(model.itervalues().next())

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.modelweight = defaultdict(
		    lambda: max_idf,
		    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
		        np.mean([self.model[w] * self.modelweight[w]
		                 for w in words if w in self.model] or
		                [np.zeros(self.dim)], axis=0)
		        for words in X
		    ])

def word2vec(tweets):
    texts = []

    for tweet in tweets:
        texts.append(tweet.split())

    return texts

def word_embeddings(tweets, embedding):
    if embedding == "word2vec":
        X = word2vec(tweets)
        w2v = Word2Vec(X, size=200, window=5, sg=0)
        model = dict(zip(w2v.wv.index2word, w2v.wv.syn0))
        
    elif embedding == "glove":
        with open("EmbeddingsGloVe/glove.twitter.27B.200d.txt", "rb") as lines:
            model = {line.split()[0]: np.array(map(float, line.split()[1:]))
                for line in lines}

    else:
        raise IOError("Dimensão do Embedding incorreta.")

    vec = TfidfEmbeddingVectorizer(model)
    vec.fit(tweets)
    matrix = vec.transform(tweets)

    return matrix

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def stratifiedSample():
    # Pega uma amostra estratificada, trainset com 10% da base original
    # Para aumentar a amostra basta reduzir o atributo test_size

    from sklearn.cross_validation import StratifiedShuffleSplit

    X = pd.read_table('Tweets/DataTrain/us/train_us_text.txt', sep="\n",engine="python-fwf")
    y = pd.read_table('Tweets/DataTrain/us/train_us_labels.txt', engine="python-fwf")

    sss = StratifiedShuffleSplit(y, 1, test_size=0.8, random_state=0)
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    for train_index, test_index in sss:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.ix[train_index], X.ix[test_index]
        y_train, y_test = y.ix[train_index], y.ix[test_index]
        print("\n")
    
    #temp = X_train.ix[:,'text']
    filename = 'Tweets/SampleTraining/TrainingSample.txt'
    
    X_train.to_csv(filename,index=False)
    filename = 'Tweets/SampleTraining/TrainingSampleLabel.txt'
    y_train.to_csv(filename,index=False)

def main():
    #stratifiedSample()

    train_text = pd.read_table('Tweets/SampleTraining/TrainingSample.txt', engine="python-fwf")
    train_labels = pd.read_table('Tweets/SampleTraining/TrainingSampleLabel.txt', engine="python-fwf")
    train_pre = pd.read_table('Tweets/SampleTraining/TrainingSamplePreprocessing.txt', engine="python-fwf")

    test_text = pd.read_table('Tweets/DataTest/us/us_test.txt', engine="python-fwf")
    test_labels = pd.read_table('Tweets/DataTest/us/us_test_labels.txt', engine="python-fwf")
    test_pre = pd.read_table('Tweets/DataTest/us/us_test_preprocessing.txt', engine="python-fwf")
    
    #trial_text = pd.read_table('Tweets/DataTrial/us_trial_text.txt', engine="python-fwf")
    #trial_labels = pd.read_table('Tweets/DataTrial/us_trial_labels.txt', engine="python-fwf")
    	
    # ======================================================================================================================= #

    train_text = train_text['text']
    train_labels = train_labels['labels']
    trainpre = train_pre['text']
    
    #trial_text = trial_text['text']
    #trial_labels = trial_labels['labels']
    
    test_text = test_text['text']
    test_labels = test_labels['labels']
    testpre = test_pre['text']

    # ====================================================================================================================== #
    	
    ''' Criação das matrizes de Word Embeddings '''

    #embedding = "word2vec"
    embedding = "glove"

    print("Criação das matrizes a partir do GloVe...")
    emb_train = word_embeddings(train_text, embedding)
    #emb_trial = word_embeddings(trial_text, embedding)
    emb_test =  word_embeddings(test_text, embedding)
    print("Criação das matrizes de Word Embeddings realizada!")
    
    # ====================================================================================================================== #

    ''' Criação do modelo BoW, houve também estudos que realizaram concatenação de BoW com Embeddings '''
    
    print("Criação do modelo BoW...")
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,4), decode_error='ignore', max_features=3500)
    bow_train = vec.fit_transform(train_text).toarray()
    #bow_trial = vec.transform(trial_text).toarray()
    bow_test =  vec.transform(test_text).toarray()
    print("Modelo BoW criado...")
    
    print("Concatenando Embeddings com BoW...")
    train = np.concatenate((emb_train, bow_train), axis=1)
    #trial = np.concatenate((emb_trial, bow_trial), axis=1)
    test = np.concatenate((emb_test, bow_test), axis=1)
    print("Concatenação realizada!")
    
    # ====================================================================================================================== #

    print("Treinando modelo...")
    #clf = LogisticRegression(C=10.0, random_state=0)
    #clf =  LinearSVC()
    clf = RandomForestClassifier()

    clf.fit(train, train_labels)
    
    print("Treinamento realizado!")
    filename = 'train.sav'
    pickle.dump(clf, open(filename, 'wb'))	
    
    print("Testando modelo...")
    prediction = clf.predict(test)
    
    # Salva as classes previstas em arquivo de txt
    
    print("Salvando as classes previstas em arquivo de txt")
    print
    prediction.dtype = np.int
    #np.savetxt('english.output.bagofwords.txt', prediction, fmt='%d')
    #np.savetxt('english.output.glove.svm.txt', prediction, fmt='%d')
    np.savetxt('english.output.glove.rf.txt', prediction, fmt='%d')
    #np.savetxt('english.output.svm.txt', prediction, fmt='%d')
    #np.savetxt('english.output.rf.txt', prediction, fmt='%d')
    
    # Código para testar train e trial
    #evaluate("us_test_labels.txt", "english.output.bagofwords.txt")
    #evaluate("us_test_labels.txt", "english.output.glove.svm.txt")
    evaluate("us_test_labels.txt", "english.output.glove.rf.txt")
    #evaluate("us_test_labels.txt", "english.output.svm.txt")
    #evaluate("us_test_labels.txt", "english.output.rf.txt")

main()