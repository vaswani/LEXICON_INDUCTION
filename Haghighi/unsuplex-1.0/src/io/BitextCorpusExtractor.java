package io;

import java.util.ArrayList;
import java.util.List;

import fig.basic.Pair;

public interface BitextCorpusExtractor {
	public Pair<List<List<String>>, List<List<String>>> extractCorpus(Pair<List<List<String>>, List<List<String>>> orig);
	
	public static class NoOpExtractor implements BitextCorpusExtractor  {
		public Pair<List<List<String>>, List<List<String>>> extractCorpus(Pair<List<List<String>>, List<List<String>>> orig) {
			return orig;
		}		
	}
	
	public static class MixedExtractor implements BitextCorpusExtractor {
		
		int numParallelSentences = 0;
		
		public MixedExtractor(int numParallelSentences) {
			this.numParallelSentences = numParallelSentences;
		}
		
		public MixedExtractor() {
			this(0);
		}

		public Pair<List<List<String>>, List<List<String>>> extractCorpus(Pair<List<List<String>>, List<List<String>>> orig) {
			// TODO Auto-generated method stub
			List<List<String>> domCorpus = new ArrayList<List<String>>();
			List<List<String>> codomCorpus = new ArrayList<List<String>>();
			
			List<List<String>> allDomCorpus = orig.getFirst();
			List<List<String>> allCodomCorpus = orig.getSecond();
			domCorpus.addAll(allDomCorpus.subList(0, numParallelSentences));
			codomCorpus.addAll(allCodomCorpus.subList(0, numParallelSentences));
			
			if (false) {
				for (int i=numParallelSentences; i < allDomCorpus.size(); ++i) {
					boolean even = i%2 == 0;
					if (even) {
						domCorpus.add(allDomCorpus.get(i));
					} else {
						codomCorpus.add(allCodomCorpus.get(i));
					}
				}
			}
			if (true) {
				int num = Math.min(allDomCorpus.size(), allCodomCorpus.size()) - numParallelSentences;
				int half = num/2 ;
				domCorpus.addAll(allDomCorpus.subList(numParallelSentences, half+numParallelSentences));
				codomCorpus.addAll(allCodomCorpus.subList(half+numParallelSentences, allCodomCorpus.size()));
			}
			return Pair.newPair(domCorpus, codomCorpus);
		}
		
		public String toString() {
			return String.format("MixedExtractor(%d)",numParallelSentences);
		}
	}
	
}
