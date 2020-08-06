import pandas as pd
import numpy as np
import pickle
import random


def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File, 'r', encoding="utf-8")
    gloveModel = {}
    for line in f:
        splitLines = line.split(' ')
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel


if __name__ == '__main__':
    glove = loadGloveModel('glove.840B.300d.txt')
    f = open('glove_dict.pkl', 'wb')
    pickle.dump(glove, f)