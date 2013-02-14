import sys

from collections import defaultdict

#first reading the hypothesis 
hyp_matching = defaultdict(lambda : defaultdict(str))
gold_matching =  defaultdict(lambda : defaultdict(int))
 
for line in open(sys.argv[1]):
    line = line.strip()
    if line == '':
        continue
    source_word,target_word,score = line.split()
    hyp_matching[source_word][target_word] = score
    


#reading the gold mapping

for line in open(sys.argv[2]):
    line = line.strip()
    if line == '' :
        continue
    #print line
    source_word,target_word = line.split(',')
    gold_matching[source_word][target_word] = 1

true_positives = 0
false_positives = 0
true_negatives=0
false_negatives=0
#computing false positives and true positives
for source_word in hyp_matching:
    if source_word in gold_matching:
        found = False
        for target_word in hyp_matching[source_word]:
            if target_word in gold_matching[source_word]:
                true_positives += 1
                found = True
                break
        if found == False:
            false_positives += 1
    else:
        false_positives += 1

#computing the false negatives
for source_word in gold_matching:
    if source_word in hyp_matching:
        found = False
        for target_word in gold_matching[source_word]:
            if target_word in hyp_matching[source_word]:
                found = True
                break
        if found == False:
            false_negatives += 1
    else:
        false_negatives += 1

precision = float(true_positives)/(true_positives+false_positives)
recall = float(true_positives)/(true_positives+false_negatives)

print 'true positives ',true_positives
print 'false positives',false_positives
print 'false negatives',false_positives
print 'precision: ',precision
print 'recall: ',recall

