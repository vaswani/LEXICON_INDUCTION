package io;

import io.POSTagPrefixes.POSTag;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import translex.FunctionalTranslationLexicon;

import fig.basic.Pair;

public class CachedBitext extends Bitext {

	private Pair<List<List<String>>,List<List<String>>> cachedCorpora = null;
	private Pair<List<List<String>>,List<List<String>>> cachedReducedCorpora = null;
	private Set<POSTag> lastTagSet = new HashSet<POSTag>();
	private Pair<Map<String,POSTag>,Map<String,POSTag>> cachedTags = null;
	private FunctionalTranslationLexicon cachedLex = null;
	private String basePath;
	private String baseFile;
	private int maxSentences;

	public CachedBitext(String basePath0, String baseFile0, int maxSentences0) {
		basePath = basePath0;
		baseFile = baseFile0;
		maxSentences = maxSentences0;
	}

	public Pair<List<List<String>>,List<List<String>>> getRawCorpora() {
			List<List<String>> domCorpus = BitextIO.readSentences(basePath+"/"+baseFile+"."+FileExtensions.DOMAIN_CORPUS_EXTENSION, maxSentences);
			List<List<String>> codomCorpus = BitextIO.readSentences(basePath+"/"+baseFile+"."+FileExtensions.CODOMAIN_CORPUS_EXTENSION, maxSentences);
			return Pair.newPair(domCorpus, codomCorpus);
	}

	// remove all words whose POS in from tag map is not in tagSet
	public Pair<List<List<String>>,List<List<String>>> getReducedCorpora(Set<POSTag> tagSet) {
		boolean sameAsLastTagSet = true;
		for (POSTag tag : tagSet) if (!lastTagSet.contains(tag)) sameAsLastTagSet = false;
		for (POSTag tag : lastTagSet) if (!tagSet.contains(tag)) sameAsLastTagSet = false;
		if (cachedReducedCorpora == null || !sameAsLastTagSet) {
			Pair<List<List<String>>,List<List<String>>> corpora = getCorpora();
			Pair<Map<String,POSTag>,Map<String,POSTag>> tags = getTags();
			List<List<String>> domReducedCorpus = reduceCorpus(corpora.getFirst(), tags.getFirst(), tagSet);
			List<List<String>> codomReducedCorpus = reduceCorpus(corpora.getSecond(), tags.getSecond(), tagSet);
			cachedReducedCorpora = Pair.newPair(domReducedCorpus, codomReducedCorpus);
		}
		return cachedReducedCorpora;
	}

	public Pair<Map<String,POSTag>,Map<String,POSTag>> getTags() {
		if (cachedTags == null) {
			cachedTags = readTags();
		}
		return cachedTags;
	}

	public FunctionalTranslationLexicon getLexicon() {
		if (cachedLex == null) {
			cachedLex = readLexicon(); 
		}
		return cachedLex;
	}

	private Pair<Map<String,POSTag>,Map<String,POSTag>> readTags() {
		Map<String,POSTag> domTags = new HashMap<String,POSTag>();
		Map<String,POSTag> codomTags = new HashMap<String,POSTag>();
		List<Pair<String,String>> domTagPairs = TextIO.readWordPairList(basePath+"/"+baseFile+"."+FileExtensions.DOMAIN_TAG_MAP_EXTENSION);
		List<Pair<String,String>> codomTagPairs = TextIO.readWordPairList(basePath+"/"+baseFile+"."+FileExtensions.CODOMAIN_TAG_MAP_EXTENSION);
		for (Pair<String,String> pair : domTagPairs) {
			domTags.put(pair.getFirst(), POSTag.valueOf(pair.getSecond()));
		}
		for (Pair<String,String> pair : codomTagPairs) {
			codomTags.put(pair.getFirst(), POSTag.valueOf(pair.getSecond()));
		}
		return Pair.newPair(domTags, codomTags);
	}

	private FunctionalTranslationLexicon readLexicon() {
		FunctionalTranslationLexicon lex = new FunctionalTranslationLexicon();
		List<Pair<String,String>> transPairs = TextIO.readWordPairList(basePath+"/"+baseFile+"."+FileExtensions.LEX_EXTENSION);
		for (Pair<String,String> trans : transPairs) {
			lex.addTranslation(trans.getFirst(), trans.getSecond());
		}
		return lex;
	}

	@Override
	public Pair<List<List<String>>, List<List<String>>> getCorpora() {
		if (cachedCorpora == null) {
			 cachedCorpora = getRawCorpora();
		}
		return cachedCorpora;
	}

}
