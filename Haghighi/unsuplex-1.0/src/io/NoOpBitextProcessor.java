package io;

import java.util.List;

//import com.sun.tools.javac.resources.compiler;

public class NoOpBitextProcessor extends InterleavedDisjointBitextProcessor {

	public NoOpBitextProcessor() {
		
	}
	
	protected List<List<String>> interleaveCorpus(List<List<String>> corpus, boolean firstHalf) {
//		List<List<String>> interCorpus = new ArrayList<List<String>>();
//		int half = corpus.size() / 2;
//		for (List<String> sent : corpus.subList(firstHalf ? 0 : half, firstHalf ? half : corpus.size())) {
//			interCorpus.add(sent);
//		}
		return corpus; //interCorpus;
	}
}
