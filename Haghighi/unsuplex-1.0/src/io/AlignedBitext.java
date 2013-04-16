package io;

import io.POSTagPrefixes.POSTag;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;


import translex.FunctionalTranslationLexicon;
import translex.ProbabilisticTranslationLexicon;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.*;

public class AlignedBitext extends CachedBitext {
	
	static final double CORRECT_ALIGN_FREQ_CUTOFF = 3;

	String basePath;
	String baseFile;
	int maxSentences;
	private Pair<List<List<String>>,List<List<String>>> cachedTagCorpora = null;
	private Set<POSTag> lastTagSet = new HashSet<POSTag>();
	private Pair<List<List<String>>,List<List<String>>> cachedReducedCorpora = null;
	private Pair<Map<String,POSTag>,Map<String,POSTag>> cachedTags = null;
	private ProbabilisticTranslationLexicon cachedProbLex = null;
	private FunctionalTranslationLexicon cachedFuncLex = null;

	public AlignedBitext(String basePath0, String baseFile0, int maxSentences0) {
		super(basePath0, baseFile0, maxSentences0);
		basePath = basePath0;
		baseFile = baseFile0;
		maxSentences = maxSentences0;
	}

	// copora with each word replaced by its POS
	public Pair<List<List<String>>,List<List<String>>> getTagCorpora() {
		if (cachedTagCorpora == null) {
			List<List<String>> domTagsCorpus = BitextIO.readSentences(basePath+"/"+baseFile+"."+FileExtensions.DOMAIN_TAG_EXTENSION, maxSentences);
			List<List<String>> codomTagsCorpus = BitextIO.readSentences(basePath+"/"+baseFile+"."+FileExtensions.CODOMAIN_TAG_EXTENSION, maxSentences);
			cachedTagCorpora = Pair.newPair(domTagsCorpus, codomTagsCorpus);
		}
		return cachedTagCorpora;
	}
	
	public Pair<Map<String,POSTag>,Map<String,POSTag>> getTags() {
		if (cachedTags == null) {
			Pair<List<List<String>>,List<List<String>>> corpora = getCorpora();
			Pair<List<List<String>>,List<List<String>>> tagCorpora = getTagCorpora();
			Map<String,POSTag> domTags = getTags(corpora.getFirst(), tagCorpora.getFirst(), true);
			Map<String,POSTag> codomTags = getTags(corpora.getSecond(), tagCorpora.getSecond(), false);
			cachedTags = Pair.newPair(domTags, codomTags);
		}
		return cachedTags;
	}
	
	private static  Map<String,POSTag> getTags(List<List<String>> corpus, List<List<String>> tagCorpus, boolean dom) {
		CounterMap<String,POSTag> tagsCounter = new CounterMap<String,POSTag>();
		for (int i=0; i<corpus.size(); i++) {
			List<String> sent = corpus.get(i);
			List<String> tagSent = tagCorpus.get(i);
			for (int j=0; j<sent.size(); j++) {
				String word = sent.get(j);
				String corpusTag = tagSent.get(j);
				tagsCounter.incrementCount(word, POSTagPrefixes.getPOSTag(corpusTag, dom), 1.0);
			}
		}
		Map<String,POSTag> tags = new HashMap<String, POSTag>();
		for (String word : tagsCounter.keySet()) {
			tags.put(word, tagsCounter.getCounter(word).argMax());
		}
		return tags;
	}
	
	// remove all words whose POS in tagCorpora is not in tagSet (different from using result of getTags for reduction)
	public Pair<List<List<String>>,List<List<String>>> getReducedCorpora(Set<POSTag> tagSet) {
		boolean sameAsLastTagSet = true;
		for (POSTag tag : tagSet) if (!lastTagSet.contains(tag)) sameAsLastTagSet = false;
		for (POSTag tag : lastTagSet) if (!tagSet.contains(tag)) sameAsLastTagSet = false;
		if (cachedReducedCorpora == null || !sameAsLastTagSet) {
			Pair<List<List<String>>,List<List<String>>> corpora = getCorpora();
			Pair<List<List<String>>,List<List<String>>> tagCorpora = getTagCorpora();
			List<List<String>> domReducedCorpus = reduceCorpus(tagSet, corpora.getFirst(), tagCorpora.getFirst(), true);
			List<List<String>> codomReducedCorpus = reduceCorpus(tagSet, corpora.getSecond(), tagCorpora.getSecond(), false);
			cachedReducedCorpora = Pair.newPair(domReducedCorpus, codomReducedCorpus);
		}
		return cachedReducedCorpora;
	}
	
	// get the correct translation lexicon based on alignments file, along with normalized alignment counts
	public ProbabilisticTranslationLexicon getProbabilisticLexicon() {
		if (cachedProbLex == null) {
			List<SentencePair> sentencePairs = BitextIO.readSentencePairs(basePath+"/"+baseFile, FileExtensions.DOMAIN_CORPUS_EXTENSION, FileExtensions.CODOMAIN_CORPUS_EXTENSION, -1);
			List<Alignment> alignments = BitextIO.readAlignments(basePath+"/"+baseFile, FileExtensions.ALIGN_EXTENSION, -1);
			cachedProbLex =  buildLexicon(sentencePairs, alignments);
		}
		return cachedProbLex;
	}

	// get the correct translation lexicon based on alginments file, reduced to be functional
	public FunctionalTranslationLexicon getLexicon() {
		if (cachedFuncLex == null) {
			ProbabilisticTranslationLexicon probLex = getProbabilisticLexicon();
			cachedFuncLex = probLex.asFunctionalTranslationLexicon(probLex.getDomain(), probLex.getCodomain());
		}
		return cachedFuncLex;
	}
	
	public void writeTags() {
		Pair<Map<String,POSTag>,Map<String,POSTag>> tags = getTags();
		writeMap(tags.getFirst(), basePath+"/"+baseFile+"."+FileExtensions.DOMAIN_TAG_MAP_EXTENSION);
		writeMap(tags.getSecond(), basePath+"/"+baseFile+"."+FileExtensions.CODOMAIN_TAG_MAP_EXTENSION);		
	}
	
	public void writeLexicon() {
		Map<String,String> lex  = getLexicon();
		writeMap(lex, basePath+"/"+baseFile+"."+FileExtensions.LEX_EXTENSION);
	}
	
	private static void writeMap(Map<? extends Object,? extends Object> map, String fileName) {
		List<Pair<String,String>> pairList = new ArrayList<Pair<String,String>>();
		for (Map.Entry<? extends Object,? extends Object> entry : map.entrySet()) {
			pairList.add(Pair.newPair(entry.getKey().toString(),entry.getValue().toString()));
		}
		TextIO.writeWordPairList(pairList, fileName);
	}
	
	// build gold lexicon given bitext and correct alignments
	private static ProbabilisticTranslationLexicon buildLexicon(List<SentencePair> sentencePairs, List<Alignment> alignments) {
		Set<String> domain = new HashSet<String>();
		Set<String> codomain = new HashSet<String>();
		CounterMap<String,String> transCounterMap0 = new CounterMap<String,String>();
		CounterMap<String,String> transCounterMap = new CounterMap<String,String>();
		for (int i = 0; i < sentencePairs.size(); i++) {
			Alignment alignment = alignments.get(i);
			SentencePair sentPair = sentencePairs.get(i);
			for (Pair<Integer,Integer> alignPair : alignment.sureAlignments) {
				String domainWord = sentPair.getDomainWords().get(alignPair.getFirst());
				String codomainWord = sentPair.getCodomainWords().get(alignPair.getSecond());
				domain.add(domainWord);
				codomain.add(codomainWord);
				transCounterMap0.incrementCount(domainWord, codomainWord, 1.0);
			}
		}
		
		for (String domainWord : transCounterMap0.keySet()) {
			Counter<String> codomainCounter = transCounterMap0.getCounter(domainWord);
			for (Entry<String,Double> entry : codomainCounter.getEntrySet()) {
				if (entry.getValue() > CORRECT_ALIGN_FREQ_CUTOFF) {
					transCounterMap.setCount(domainWord, entry.getKey(), entry.getValue());
				}
			}
		}
		transCounterMap.normalize();
		return new ProbabilisticTranslationLexicon(domain, codomain, transCounterMap);
	}
	
	public static void main(String[] args) {
		AlignedBitext bitext = new AlignedBitext(args[0], args[1], -1);		
		bitext.writeTags();
		//bitext.writeLexicon();
	}
	
}
