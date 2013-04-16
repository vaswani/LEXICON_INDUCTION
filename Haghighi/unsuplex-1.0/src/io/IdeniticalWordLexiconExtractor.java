package io;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import fig.basic.Pair;

public class IdeniticalWordLexiconExtractor {
	
	public static void main(String[] args) {
		
		String basePath = args[0];
		String file1 = args[1];
		String file2 = args[2];
		String outFile = args[3];
		
		List<List<String>> corpus1 = TextIO.readSentences(basePath+"/"+file1, -1);
		List<List<String>> corpus2 = TextIO.readSentences(basePath+"/"+file2, -1);
		
		Set<String> words = new HashSet<String>();
		for (List<String> sent : corpus1) words.addAll(sent);
		for (List<String> sent : corpus2) words.addAll(sent);
		System.out.printf("Num total word types: %d", words.size());
		
		
		List<Pair<String,String>> lex = new ArrayList<Pair<String,String>>();
		for (String word : words) lex.add(Pair.newPair(word, word));
		
		TextIO.writeWordPairList(lex, basePath+"/"+outFile);
		
	}

}
