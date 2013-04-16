package io;

import io.POSTagPrefixes.POSTag;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import translex.FunctionalTranslationLexicon;
import fig.basic.Pair;

public class MixedBitext implements BitextProcessor {
	
	private Bitext bitext;
	private Pair<List<List<String>>,List<List<String>>> cachedCorpora = null;
	private Pair<List<List<String>>,List<List<String>>> cachedReducedCorpora = null;
	private Set<POSTag> tagSet = new HashSet<POSTag>();
	private Set<POSTag> lastTagSet = new HashSet<POSTag>();
	private int numParallelSentences = 0;
	
	public MixedBitext() {
		
	}
	
	
	public String toString() {
		return String.format("MixedBitext(%d)", numParallelSentences);
	}
	
	public void setNumParallelSentences(int numParallelSentences) {
		this.numParallelSentences = numParallelSentences;
	}
	
	public Pair<List<List<String>>, List<List<String>>> getCorpora() {
		if (cachedCorpora == null) {
			List<List<String>> domCorpus = new ArrayList<List<String>>();
			List<List<String>> codomCorpus = new ArrayList<List<String>>();
			
			Pair<List<List<String>>,List<List<String>>> corpora = bitext.getCorpora();
			List<List<String>> allDomCorpus = corpora.getFirst();
			List<List<String>> allCodomCorpus = corpora.getSecond();
			domCorpus.addAll(allDomCorpus.subList(0, numParallelSentences));
			codomCorpus.addAll(allCodomCorpus.subList(0, numParallelSentences));
					
			int n = allDomCorpus.size()-numParallelSentences;
			int half = n / 2;
			List<List<String>> restDomCorpus = allDomCorpus.subList(numParallelSentences, numParallelSentences+half);
			List<List<String>> restCodomCorpus = allCodomCorpus.subList(numParallelSentences+half, allCodomCorpus.size());
			
			domCorpus.addAll(restDomCorpus);
			codomCorpus.addAll(restCodomCorpus);
			
			cachedCorpora =  Pair.newPair(domCorpus, codomCorpus);
		}
		return cachedCorpora;
	}
	
	

	public Pair<List<List<String>>, List<List<String>>> getReducedCorpora(Set<POSTag> tagSet) {
		boolean sameAsLastTagSet = tagSet.equals(lastTagSet) && (!tagSet.isEmpty() || cachedReducedCorpora != null);
		if (!sameAsLastTagSet && cachedReducedCorpora == null) {
			Pair<List<List<String>>,List<List<String>>> corpora = getCorpora();
			Pair<Map<String,POSTag>,Map<String,POSTag>> tags = getTags();
			List<List<String>> domReducedCorpus = Bitext.reduceCorpus(corpora.getFirst(), tags.getFirst(), tagSet);
			List<List<String>> codomReducedCorpus = Bitext.reduceCorpus(corpora.getSecond(), tags.getSecond(), tagSet);
			cachedReducedCorpora = Pair.newPair(domReducedCorpus, codomReducedCorpus);
		}
		return cachedReducedCorpora;
	}

	public FunctionalTranslationLexicon getLexicon() {
		return bitext.getLexicon();
	}
	
	public Pair<Map<String, POSTag>, Map<String, POSTag>> getTags() {
		return bitext.getTags();
	}
	
	protected List<List<String>> interleaveCorpus(List<List<String>> corpus, boolean odd) {
		List<List<String>> interCorpus = new ArrayList<List<String>>();
		for (int i = 0; i < corpus.size(); i++) {
			int parity = i % 2;
			if ((odd && parity == 1) || (!odd && parity == 0)) {
				interCorpus.add(corpus.get(i));
			}
		}
		return interCorpus;
	}

	public Pair<List<List<String>>, List<List<String>>> getReducedCorproa() {
		// TODO Auto-generated method stub
		return getReducedCorpora(tagSet);
	}

	public void setBitext(Bitext bitext) {
		// TODO Auto-generated method stub
		this.bitext = bitext;
	}

	public void setPOSTags(Set<POSTag> posTags) {
		this.tagSet = posTags;
		
	}

}
