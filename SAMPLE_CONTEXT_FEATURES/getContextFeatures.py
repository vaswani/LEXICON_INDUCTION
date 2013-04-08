import sys

tag_list = []
word_list = []
from collections import defaultdict

#reading the tags
for line in open(sys.argv[1]):
    line = line.strip();
    if line == '':
        continue
    #print line

    templist = line.split()
    lowercased_word = templist[len(templist)-1]
    tag =  templist[len(templist)-2]
    word = '_'.join(templist[0:len(templist)-2])
    word = word.replace("-","_")

    
    word_list.append(word.lower())
    #rest as is    
    tag_list.append(tag)

counter = 0

data_tag_list = []
data_word_list = []


'''
for line_num,line in enumerate(open(sys.argv[2])):
    line = line.strip();
    if line == '':
        continue
    words = line.split()
    tags = []
    data_word_list.append(words)
    for i,word in enumerate(words):
        if words[i] != word_list[counter+i]:
            print words[i],' in the original file was not found with a tag in the tags file and the counter is ',counter+i
            sys.exit()
    counter += len(words)
    
'''
noun_tags = {}
if sys.argv[3] == 'en':
    # Penntreebank tags from tree tagger are different
    # noun_tags = {'NN':1,'NNS':1,'NNP':1,'NNPS':1} 
    noun_tags = {'NN':1,'NNS':1,'NP':1,'NPS':1}
elif sys.argv[3] == 'es':
    noun_tags = {'NMEA':1,'NC':1,'NP':1,'NMON':1}
elif sys.argv[3] == 'fr':
    noun_tags = {'NAM':1,'NOM':1}

#nouns = dict((line.strip(),1) for line in open(sys.argv[2]))
nouns_dict = dict()
frequency = dict()
for line in open(sys.argv[2]):
    temp = line.strip().split()
    if(len(temp) < 2):
        print 'error in freq noun file'
    elif(len(temp) == 2):
        word = temp[1]
        word = word.replace("-","_")
        nouns_dict[word] = 1
        frequency_value = temp[0]
        frequency[word] = frequency_value
    else:
        word = '_'.join(temp[1:]);
        word = word.replace("-","_")
        nouns_dict[word] = 1
        frequency_value = temp[0]
        frequency[word] = frequency_value


nouns = nouns_dict.keys()
#print 'the size of nouns is ',len(nouns)
#raw_input()
#print len(nouns)
feature_window_size = int(sys.argv[4])
#print 'nouns is ',nouns
#getting noun context features
#print 'building context features'
if sys.argv[5] != 'DIRECTIONAL':
    context_features = defaultdict(lambda:defaultdict(int))
    for i,word in enumerate(word_list) :
        if word in nouns and tag_list[i] in noun_tags:
            for j in range(-feature_window_size,feature_window_size+1):
                if i+j < 0 or j==0 or i+j >= len(word_list):
                    continue
                if tag_list[i+j] in noun_tags and word_list[i+j] in nouns_dict:
                    context_features[word][word_list[i+j]] += 1
    #print 'built the context features for the nouns'
    for noun in nouns:
        #print 'the size of the feature list for nouns is ',len(context_features[noun])
        #raw_input()
        print "%s %s %s"%('_' + noun + '_',frequency[noun],' '.join(repr(context_features[noun][word]) for word in nouns))
else:
    left_context_features = defaultdict(lambda:defaultdict(int))
    right_context_features = defaultdict(lambda:defaultdict(int))
    for i,word in enumerate(word_list) :
        if word in nouns and tag_list[i] in noun_tags:
            for j in range(-feature_window_size,0):
                if i+j < 0:
                    continue
                if tag_list[i+j] in noun_tags and word_list[i+j] in nouns_dict:
                    left_context_features[word][word_list[i+j]] += 1
            for j in range(1,feature_window_size+1):
                if i+j >= len(word_list):
                    continue
                if tag_list[i+j] in noun_tags and word_list[i+j] in nouns_dict:
                    right_context_features[word][word_list[i+j]] += 1

    #print 'built the context features for the nouns'
    for noun in nouns:
        #print 'the size of the feature list for nouns is ',len(context_features[noun])
        #raw_input()
        print "%s %s %s %s"%('_' + noun + '_',frequency[noun],' '.join(repr(left_context_features[noun][word]) for word in nouns),' '.join(repr(right_context_features[noun][word]) for word in nouns))

        #print "%s %s"%('_' + noun + '_',' '.join(repr(context_features[noun][word]) for word in nouns))
