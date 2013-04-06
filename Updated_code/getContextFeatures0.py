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

    for pair in templist:
        if sys.argv[3] == 'ch':
            (word,tag)= pair.split('#')
        elif sys.argv[3] == 'ar':
            arr = pair.split('/')
            if (len(arr) == 2):
                (word,tag)= pair.split('/')
            elif (len(arr) > 2):
                tag = arr[len(arr)-1]
                word = '/'.join(arr[0:len(arr)-1])

        
        word_list.append(word.lower())
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
if sys.argv[3] == 'ch':
    noun_tags = {'NR':1,'NT':1,'NN':1}
elif sys.argv[3] == 'ar':
    noun_tags = {'NN':1,'NP':1,'NNS':1,'NPS':1,'DTNN':1,'DTNP':1,'DTNNS':1,'DTNPS':1}

#nouns = dict((line.strip(),1) for line in open(sys.argv[2]))
nouns = dict()
for line in open(sys.argv[2]):
    temp = line.strip().split()
    if(len(temp) < 2):
        print 'error in freq noun file'
    elif(len(temp) == 2):
        nouns[temp[1]] = 1
    else:
        nouns['_'.join(temp[1:])] = 1

#print 'the size of nouns is ',len(nouns)
#raw_input()
#print len(nouns)

context_features = defaultdict(lambda:defaultdict(int))
#print 'nouns is ',nouns
#getting noun context features
#print 'building context features'
for i,word in enumerate(word_list) :
    if word in nouns and tag_list[i] in noun_tags:
        #looking at the left window
        for j,window in enumerate(word_list[max(0,i-4):i]):
            if tag_list[max(0,i-(4-j))] in noun_tags and word_list[max(0,i-(4-j))] in nouns:
                context_features[word][window] += 1     
        #looking at the right window
        for j,window in enumerate(word_list[i+1:i+5]):
            if tag_list[i+1+j] in noun_tags and  word_list[i+1+j] in nouns:
                context_features[word][window] += 1     

#print 'built the context features for the nouns'

for noun in nouns:
    #print 'the size of the feature list for nouns is ',len(context_features[noun])
    #raw_input()
    print "%s %s"%('_' + noun + '_',' '.join(repr(context_features[noun][word]) for word in nouns))
