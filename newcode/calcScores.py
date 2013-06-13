# computes precision and recall scores
import sys
from collections import defaultdict


# dictionary contains the list of all matching obtained from the gold lexicon.
# dictionary has format key, key of values.
# source words is a dictionary of source words and target words is a dictionary of target words
def filterDictionary(dictionary, source_words, target_words):
    filtered_dictionary = defaultict(defaultdict(string))
    total_items = 0
    for source_word in dictionary:
        if source_word not in source_words:
            continue
        for matched_target_word in dictionary[source_word]:
            if matched_target_word in target_words:
               filtered_dictionary[source_word][matched_target_word] = 1
               total_items +=1
    return filtered_dictionary,total_items


# filtered dictionary is a dictionary with the key as a source word and the value
# is another dictionary with the key as the target words
# matches is a list of lists. Each row in the list has the format
# [source_word,matched_target_word,score]. The items are sorted according to score
def computePrecisionRecall(filtered_dictionary, matches, dictionary_size, beta):
    recall_cutoffs=[0.05, 0.1, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6]
    beta_squared = beta*beta
    recall_cutoff_index = 0
    num_matches = 0.
    for i, match in enumerate(matches):
        if match[0] in filtered_dictionary:
            if match[1] in filtered_dictionary[match[0]] :
                num_matches += 1
                recall = num_matches/dictionary_size
                #print recall if its crossed a cutoff
                if recall >= recall_cutoffs[recall_cutoff_index]: 
                    precision = num_matches/i+1
                    F1 = (1+beta_squared)*precision*recall/(beta_squared*precision+recall)
                    print 'precision at recall ', recall_cutoffs[recall_cutoff_index], \
                        ' is ', repr(precision), ' and the F1 is ', repr(F1)
                    recall_cutoff_index += 1


if __name__ == '__main__':
    print 1
