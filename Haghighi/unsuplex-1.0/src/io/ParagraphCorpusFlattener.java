package io;

import java.util.ArrayList;
import java.util.List;

public class ParagraphCorpusFlattener {
	
	public static void main(String[] args) {
		
		String basePath = args[0];
		String file = args[1];
		
		List<List<String>> corpus = TextIO.readSentences(basePath+"/"+file, -1);
		List<List<String>> newCorpus = new ArrayList<List<String>>();

		for (List<String> sent : corpus) {
			List<String> newSent = new ArrayList<String>();
			for (String word : sent) {
				newSent.add(word);
				if (word.equals(".")) {
					newCorpus.add(newSent);
					newSent = new ArrayList<String>();
				}
			}
			if (newSent.size() != 0) {
				newCorpus.add(newSent);
			}
		}
		
		TextIO.writeSentences(newCorpus, basePath+"/"+file);
	}

}
