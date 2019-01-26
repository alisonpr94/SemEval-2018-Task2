# -*- coding: utf-8 -*-

#@author: Alison Ribeiro

import sys
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def evaluate(path_real, path_predict):
    real = []
    predict = []

    file = open(path_real)
    for row in file.readlines():
        clean = row.strip('\n')
        real.append(int(clean))
    file.close()

    file = open(path_predict)
    for row in file.readlines():
        clean = row.strip('\n')
        predict.append(int(clean))
    file.close()

    if len(predict)!=len(real): sys.exit('ERROR: O número de arquivos é diferente.')

    print("F1.........: %.3f" %(f1_score(real, predict, average="macro") * 100))
    print("Precision..: %.3f" %(precision_score(real, predict, average="macro") * 100))
    print("Recall.....: %.3f" %(recall_score(real, predict, average="macro") * 100))
    print("Accuracy...: %.3f" %(accuracy_score(real, predict) * 100))
