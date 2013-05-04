package io;

import io.POSTagPrefixes.POSTag;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import translex.FunctionalTranslationLexicon;
import edu.berkeley.nlp.util.Counter;
import fig.basic.Indexer;
import fig.basic.Pair;

public abstract class Bitext {

	private Pair<Counter<String>,Counter<String>> cachedWordCounts = null;
	private Pair<List<List<String>>, List<List<String>>> cachedCorpora = null;
	private Pair<Map<String, POSTag>, Map<String, POSTag>> cachedTags = null;
	private Pair<List<List<String>>, List<List<String>>> cachedReducedCorpora = null;
	private BitextCorpusExtractor bitextCorpusExtractor;
	private Set<POSTag> lastTagSet = new HashSet<POSTag>();
//	final private String basePath ;
//	final private String baseFile ;
//	final int maxSentences ;

//	public Bitext(String basePath0, String baseFile0, int maxSentences0) {
//		basePath = basePath0;
//		baseFile = baseFile0;
//		maxSentences = maxSentences0;
//	}


	public abstract Pair<List<List<String>>, List<List<String>>> getCorpora();
//	{
//		if (cachedCorpora == null) {
//			cachedCorpora = bitextCorpusExtractor.extractCorpus(this);
//		}
//		return cachedCorpora;
//	}

//	public Pair<List<List<String>>,List<List<String>>> getRawCorpora() {
//		List<List<String>> domCorpus = BitextIO.readSentences(basePath+"/"+baseFile+"."+FileExtensions.DOMAIN_CORPUS_EXTENSION, maxSentences);
//		List<List<String>> codomCorpus = BitextIO.readSentences(basePath+"/"+baseFile+"."+FileExtensions.CODOMAIN_CORPUS_EXTENSION, maxSentences);
//		return Pair.newPair(domCorpus, codomCorpus);
//	}

//	public void setBitextCorpusExtracor(BitextCorpusExtractor bitextCorpusExtractor) {
//		this.bitextCorpusExtractor = bitextCorpusExtractor;
//	}

	public  Pair<List<List<String>>, List<List<String>>> getReducedCorpora(Set<POSTag> tagSet) {
		if (tagSet.isEmpty()) { return getCorpora(); }

		if (!lastTagSet.equals(tagSet)) {
			Pair<List<List<String>>, List<List<String>>> corpus = getCorpora();			
			List<List<String>> reducedDomain = reduceCorpus(corpus.getFirst(), getTags().getFirst(), tagSet);
			List<List<String>> reducedCodomain = reduceCorpus(corpus.getSecond(), getTags().getSecond(), tagSet);
			cachedReducedCorpora = Pair.newPair(reducedDomain, reducedCodomain);
		}
		return cachedReducedCorpora;
	}


//	private Pair<Map<String,POSTag>,Map<String,POSTag>> readTags() 
//	{
//		Map<String,POSTag> domTags = new HashMap<String,POSTag>();
//		Map<String,POSTag> codomTags = new HashMap<String,POSTag>();
//		List<Pair<String,String>> domTagPairs = TextIO.readWordPairList(basePath+"/"+baseFile+"."+FileExtensions.DOMAIN_TAG_MAP_EXTENSION);
//		List<Pair<String,String>> codomTagPairs = TextIO.readWordPairList(basePath+"/"+baseFile+"."+FileExtensions.CODOMAIN_TAG_MAP_EXTENSION);
//		for (Pair<String,String> pair : domTagPairs) {
//			domTags.put(pair.getFirst(), POSTag.valueOf(pair.getSecond()));
//		}
//		for (Pair<String,String> pair : codomTagPairs) {
//			codomTags.put(pair.getFirst(), POSTag.valueOf(pair.getSecond()));
//		}
//		return Pair.newPair(domTags, codomTags);
//	}


	public abstract Pair<Map<String, POSTag>, Map<String, POSTag>> getTags() ;
//	{
//		if (cachedTags == null) {
//			cachedTags = readTags();
//		}
//		return cachedTags;
//	}

	public abstract FunctionalTranslationLexicon getLexicon();

	public Pair<Counter<String>,Counter<String>> getWordCounts() {
		if (cachedWordCounts == null) {
			Pair<List<List<String>>,List<List<String>>> corpora = getCorpora();
			Counter<String> domCounts = new Counter<String>();
			Counter<String> codomCounts = new Counter<String>();
			for (List<String> sent : corpora.getFirst()) domCounts.incrementAll(sent, 1.0);
			for (List<String> sent : corpora.getSecond()) codomCounts.incrementAll(sent, 1.0);
			cachedWordCounts = Pair.newPair(domCounts, codomCounts);
		}
		return cachedWordCounts;
	}

	public Pair<Indexer<String>,Indexer<String>> getMostCommonWordsInOrder(Set<POSTag> tagSet, int numWords) {
		Pair<Counter<String>,Counter<String>> wordCounts = getWordCounts();
		Counter<String> domCounter = new Counter<String>(wordCounts.getFirst());
		Counter<String> codomCounter = new Counter<String>(wordCounts.getSecond());
		Indexer<String> domWords = new Indexer<String>();
		Indexer<String> codomWords = new Indexer<String>();
		Pair<Map<String,POSTag>,Map<String,POSTag>> tags = getTags();

		int domCount = 0;
		while (domCount < numWords && !domCounter.isEmpty()) {
			String domWord = domCounter.argMax();
			if (tagSet.contains(tags.getFirst().get(domWord))) {
				domWords.add(domWord);
				domCount++;
			}
			domCounter.removeKey(domWord);
		}

		int codomCount = 0;
		while (codomCount < numWords && !codomCounter.isEmpty()) {
			String codomWord = codomCounter.argMax();
			if (tagSet.contains(tags.getSecond().get(codomWord))) {
				codomWords.add(codomWord);
				codomCount++;
			}
			codomCounter.removeKey(codomWord);
		}

		return Pair.newPair(domWords, codomWords);
	}

	// remove all words whose POS in tagCorpora is not in tagSet (different from using result of getTags for reduction)
	public static List<List<String>> reduceCorpus(List<List<String>> corpus, Map<String, POSTag> tags, Set<POSTag> tagSet) {
		List<List<String>> reducedCorpus = new ArrayList<List<String>>();
		for (int i = 0; i < corpus.size(); i++) {
			List<String> reducedSent = new ArrayList<String>();
			List<String> wordSent = corpus.get(i);
			for (int j = 0; j < wordSent.size(); j++) {
				String word = wordSent.get(j);
				POSTag wordTag = tags.get(word);
				for (POSTag tag : tagSet) {
					if (tag == wordTag) {
						reducedSent.add(word);
						break;
					}
				}
			}
			if (!reducedSent.isEmpty())
				reducedCorpus.add(reducedSent);
		}
		return reducedCorpus;
	}

	// remove all words from corpus that have POS in tagSet
	protected static List<List<String>> reduceCorpus(Set<POSTag> tagSet, List<List<String>> corpus, List<List<String>> tagCorpus, boolean dom) {
		List<List<String>> reducedCorpus = new ArrayList<List<String>>();
		for (int i = 0; i < corpus.size(); i++) {
			List<String> reducedSent = new ArrayList<String>();
			List<String> wordSent = corpus.get(i);
			List<String> tagSent = tagCorpus.get(i);
			for (int j = 0; j < wordSent.size(); j++) {
				String word = wordSent.get(j);
				String corpusTag = tagSent.get(j);
				for (POSTag tag : tagSet) {
					if (POSTagPrefixes.corpusTagMatch(corpusTag, tag, dom)) {
						reducedSent.add(word);
						break;
					}
				}
			}
			if (!reducedSent.isEmpty())
				reducedCorpus.add(reducedSent);
		}
		return reducedCorpus;
	}

}