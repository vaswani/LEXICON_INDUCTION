package io;

import io.POSTagPrefixes.POSTag;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import translex.FunctionalTranslationLexicon;
import fig.basic.Pair;

public class InterleavedDisjointBitextProcessor implements BitextProcessor {
	
	private Bitext bitext;
	private Pair<List<List<String>>,List<List<String>>> cachedCorpora = null;
	private Pair<List<List<String>>,List<List<String>>> cachedReducedCorpora = null;
	private Set<POSTag> lastTagSet = new HashSet<POSTag>();
	
	public Pair<List<List<String>>, List<List<String>>> getCorpora() {
		if (cachedCorpora == null) {
			Pair<List<List<String>>,List<List<String>>> corpora = bitext.getCorpora();
			List<List<String>> domCorpus = interleaveCorpus(corpora.getFirst(), true);
			List<List<String>> codomCorpus = interleaveCorpus(corpora.getSecond(), false);
			cachedCorpora =  Pair.newPair(domCorpus, codomCorpus);
		}
		return cachedCorpora;
	}

	public Pair<List<List<String>>, List<List<String>>> getReducedCorpora(Set<POSTag> tagSet) {
		boolean sameAsLastTagSet = true;
		for (POSTag tag : tagSet) if (!lastTagSet.contains(tag)) sameAsLastTagSet = false;
		for (POSTag tag : lastTagSet) if (!tagSet.contains(tag)) sameAsLastTagSet = false;
		if (cachedReducedCorpora == null || !sameAsLastTagSet) {
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
		if (cachedReducedCorpora == null) {
			Pair<List<List<String>>,List<List<String>>> corpora = getCorpora();
			Pair<Map<String,POSTag>,Map<String,POSTag>> tags = getTags();
			List<List<String>> domReducedCorpus = Bitext.reduceCorpus(corpora.getFirst(), tags.getFirst(), lastTagSet);
			List<List<String>> codomReducedCorpus = Bitext.reduceCorpus(corpora.getSecond(), tags.getSecond(), lastTagSet);
			cachedReducedCorpora = Pair.newPair(domReducedCorpus, codomReducedCorpus);		
		}
		return cachedReducedCorpora;
	}

	public void setBitext(Bitext bitext) {
		// TODO Auto-generated method stub
		this.bitext = bitext;
	}

	public void setPOSTags(Set<POSTag> posTags) {
		// TODO Auto-generated method stub
		this.lastTagSet = posTags;
	}

}
