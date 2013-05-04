import sys
import numpy
from collections import defaultdict

#read the features in 

def readFeatures(filename):
    word_list  = []
    feature_list = []
    for line in open(filename):
        line = line.strip();
        if line == '':
            continue
        word,rest = line.split(' ',1)
        word_list.append(word)
        feature_list.append(map(float,rest.split()))
    return(word_list,numpy.array(feature_list))

def readGraph(filename,word_to_index):
    graph = numpy.zeros((len(word_to_index),len(word_to_index)),int)
    for line in open(filename):
        line = line.strip()
        if line == "":
            continue
        w1,w2,cost = line.split()
        #print w1,w2
        #raw_input()
        graph[word_to_index[w1]][word_to_index[w2]] = 1

    return graph
        
