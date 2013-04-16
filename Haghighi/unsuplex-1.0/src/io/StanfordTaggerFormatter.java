package io;

import java.util.ArrayList;
import java.util.List;

public class StanfordTaggerFormatter {
	
	public static void main(String[] args) {
		
		String basePath = args[0];
		String file = args[1];
		
		List<List<String>> tagCorpus = TextIO.readSentences(basePath+"/"+file, -1);
		List<List<String>> newTagCorpus = new ArrayList<List<String>>();

		for (List<String> sent : tagCorpus) {
			List<String> newSent = new ArrayList<String>();
			for (String word : sent) {
				int index = word.lastIndexOf("/");
				String newWord = word.substring(index+1);
				newSent.add(newWord);
			}
			newTagCorpus.add(newSent);
		}
		
		TextIO.writeSentences(newTagCorpus, basePath+"/"+file);
	}

}
