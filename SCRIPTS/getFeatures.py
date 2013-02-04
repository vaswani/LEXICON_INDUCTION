from collections import defaultdict
import sys
data = []
freq = {}
ortho_features = {}
word_list = []
ortho_dict = defaultdict(int)
for line in open(sys.argv[1]):
    line = line.strip()
    frequency,word = line.split()
    freq[word] = frequency
    word_list.append(word)

#now getting all the triple features
for word in word_list:
    for size in range(1,4):
        for index in range(0,len(word)-size+1):
            #print word[index:index+size]
            ortho_dict[word[index:index+size]] += 1

#getting the feature counts for every word

ortho_list = ortho_dict.keys()
ortho_to_int = dict((ortho,i) for i,ortho in enumerate(ortho_list))
for word in word_list:
    word_ortho_list = [0]*len(ortho_list)
    for size in range(1,4):
        for index in range(0,len(word)-size+1):
            #print word[index:index+size]
            word_ortho_list[ortho_to_int[word[index:index+size]]] += 1
        #now printing out the feature dict for the word
        print "%s %s %s"%(word,freq[word],' '.join(map(repr,word_ortho_list)))
        #raw_input()

'''
#print 'the size of the dictionary is ',len(ortho_dict)
#print ortho_dict
ortho_list = ortho_dict.keys()
for word in word_list:
    word_feature_list = []
    for feature in ortho_list:
        if feature in word:
            word_feature_list.append(1)
        else:
            word_feature_list.append(0)
    print "%s %s %s"%(word,freq[word],' '.join(map(repr,word_feature_list)))
'''    
