import sys
import cPickle as pickle
from priority_dict import *
import numpy


#embeddings_dictionary = pickle.load(open(sys.argv[1]))
embeddings_dictionary = dict((line.strip().split()[0],numpy.array(map(float,line.strip().split()[1:])))for line in open(sys.argv[1]))
#print embeddings_dictionary
word = sys.argv[2]
k = int(sys.argv[3])
if word not in embeddings_dictionary:
    print 'not in embeddings_dictionary'
    sys.exit()
word_embeddings = embeddings_dictionary[word]
distances = {}
for i in embeddings_dictionary:
    if i == word:
        continue
    distances[i] = numpy.square(word_embeddings-embeddings_dictionary[i]).sum()
distances_heap = priority_dict(distances)


for i in range(k):
    p_argmin,p_min = distances_heap.pop_smallest()
    print p_argmin,', distance:',p_min

