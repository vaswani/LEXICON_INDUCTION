import sys
import IO
import BilexiconUtil

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

    lex = BilexiconUtil.readLexicon(filename_lexicon)
    (gold_lex, times) = BilexiconUtil.filterLexicon(lex, wordsX.words, wordsY.words)
    print >> sys.stderr, 'Done filtering gold lexicon'
    seed = []
    used_targets = set()
    for source_word in wordsX.words:
        if source_word in gold_lex:
            translations = gold_lex[source_word]
            for translation in translations:
                if translation in wordsY.words and translation not in used_targets:
                    seed.append((source_word, translation))
                    used_targets.add(translation)
                    print "%s,%s" % (source_word, translation)
                    break
        else:
            pass  #print source_word, '__NA__'

        if len(seed) == Nseed:
            break

    #print seed



