import sys
import numpy
from priority_dict import *

nouns = []
embeddings_dictionary = {}

for line in open(sys.argv[1]):
    line = line.strip()
    frequency,word = line.split()
    nouns.append(word)
    #print word

#print 'the size of nouns is ',len(nouns)
for line in open(sys.argv[2]):
    #print line
    noun,embeddings = line.strip().split('\t',1)
    #print 'noun ',noun
    numpy_embeddings = numpy.array(map(float,embeddings.strip().split())) 
    if noun in nouns:
        embeddings_dictionary[noun] = numpy_embeddings
        #print 'yes'
    else:
        print 'not:',noun
        sys.exit()

#sys.exit()
k = int(sys.argv[3])

#print 'size embeddings dictionary',len(embeddings_dictionary)
#raw_input()
#first create the heap dictionary 

all_pair_distances = {}
for word in embeddings_dictionary:
    distances = {}
    for i in embeddings_dictionary:
        #print 'word i',i
        if i == word:
            continue
        distances[i] = numpy.square(embeddings_dictionary[word]-embeddings_dictionary[i]).sum()
        #print distances[i]
    #print 'distances',len(distances)
    distances_heap = priority_dict(distances)

    for i in range(k):
        p_argmin,p_min = distances_heap.pop_smallest()
        #print word,',',p_argmin,',',p_min
        print "%s\t%s\t%s"%(word,p_argmin,repr(p_min))
    
