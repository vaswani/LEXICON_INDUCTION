import numpy
from collections import defaultdict
import sys
data = []
freq = {}
ortho_features = {}
word_list = []
ortho_dict = defaultdict(int)
nouns = []
for line in open(sys.argv[1]):
    line = line.strip()
    frequency,word = line.split()
    freq[word] = frequency
    word_list.append(word)
embeddings_dictionary = {}

#print 'the size of nouns is ',len(nouns)
for line in open(sys.argv[2]):
    #print line
    noun,embeddings = line.strip().split('\t',1)
    #print 'noun ',noun
    numpy_embeddings = numpy.array(map(float,embeddings.strip().split())) 
    if noun in word_list:
        embeddings_dictionary[noun] = numpy_embeddings
    if noun == '<unk>':
        embeddings_dictionary[noun] = numpy_embeddings
for word in word_list:
    if word in embeddings_dictionary:
        print "%s %s %s"%(word,freq[word],' '.join(map(repr,embeddings_dictionary[word])))
    else :
        print "%s %s %s"%(word,freq[word],' '.join(map(repr,embeddings_dictionary['<unk>'])))


