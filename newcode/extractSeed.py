import sys
import IO
import BilexiconUtil
from common import *


if __name__ == '__main__':
    filename_wordsX = sys.argv[1]
    filename_wordsY = sys.argv[2]
    filename_lexicon = sys.argv[3]
    Nseed = int(sys.argv[4])

    if 'pickle' in filename_wordsX:
        wordsX = IO.readPickledWords(filename_wordsX)
        wordsY = IO.readPickledWords(filename_wordsY)
    else:
        wordsX = IO.readWords(filename_wordsX)
        wordsY = IO.readWords(filename_wordsY)

    if filename_lexicon == 'None':  # we don't have a lexicon. assume identity.
        log(100, 'Using identity lexicon')
        lex = None
        gold_lex = dict()  #
        for w in wordsX.words:
            gold_lex[w] = [w]
        log(100,  gold_lex)
    else:
        lex = BilexiconUtil.readLexicon(filename_lexicon)
        (gold_lex, times) = BilexiconUtil.filterLexicon(lex, wordsX.words, wordsY.words)
        log(100, 'Done filtering gold lexicon')

    seed = []
    used_targets = set()
    for source_word in wordsX.words:                      # go over source
        if source_word in gold_lex:                       # check source in lexicon
            translations = gold_lex[source_word]          # get translations of source
            for translation in translations:              # then, append translations for non-translated sources.
                if translation in wordsY.words and translation not in used_targets:
                    seed.append((source_word, translation))
                    used_targets.add(translation)
                    print "%s,%s" % (source_word, translation)
                    break
        else:
            pass  #print source_word, '__NA__'

        if len(seed) == Nseed:
            break

   # print seed
