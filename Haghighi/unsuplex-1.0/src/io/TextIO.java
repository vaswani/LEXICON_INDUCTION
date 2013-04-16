package io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import fig.basic.Pair;



public class TextIO {
	
    static final String INPUT_ENC = "UTF-8";
    static final String OUTPUT_ENC = "UTF-8";
	static final String NULL_WORD = Interners.stringInterner.intern("NULL_WORD");
	
	public static void writeWordPairList(List<Pair<String,String>> pairList, String fileName) {
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), OUTPUT_ENC));
			for (Pair<String,String> wordPair : pairList) {
				out.write(wordPair.getFirst() + " " + wordPair.getSecond());
				out.newLine();
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static List<Pair<String,String>> readWordPairList(String fileName) {
		List<Pair<String,String>> wordPairs = new ArrayList<Pair<String,String>>();
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), INPUT_ENC));
			while (in.ready()) {
				String wordPairLine = in.readLine();
				String[] wordPair = wordPairLine.split("\\s+");
				if (wordPair.length == 2 && !wordPair[0].matches("\\s*") && !wordPair[1].matches("\\s*")) {
					wordPairs.add(new Pair<String,String>(Interners.stringInterner.intern(wordPair[0]),Interners.stringInterner.intern(wordPair[1])));
				} else {
					throw new RuntimeException("Problem reading word pair line: " + wordPairLine);
				}
			}
//			System.out.printf("Read %d word pairs.%n", wordPairs.size());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return wordPairs;
	}
	
	public static void writeWordList(List<String> wordList, String fileName) {
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), OUTPUT_ENC));
			for (String word : wordList) {
				out.write(word);
				out.newLine();
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static List<String> readWordList(String fileName) {
		List<String> words = new ArrayList<String>();
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), INPUT_ENC));
			while (in.ready()) {
				String word = in.readLine();
				word = word.trim();
				if (!word.matches("\\s*"))
					words.add(Interners.stringInterner.intern(word));
			}
//			System.out.printf("Read %d words.%n", words.size());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return words;
	}
	
	public static void writeSentences(List<List<String>> sentences, String fileName) {
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), OUTPUT_ENC));
			for (List<String> sentence : sentences) {
				writeSentence(out, sentence);
				out.newLine();
				out.flush();
			}
			out.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}
	
	private static void writeSentence(BufferedWriter out, List<String> sent) throws IOException {
		for (int i=0; i<sent.size(); i++) {
			String word = sent.get(i);
			if (i==0) {
				out.write(word);
			} else {
				out.write(" "+word);
			}
		}
	}
	
	public static List<List<String>> readSentences(String fileName, int maxSentences) {
		List<List<String>> sentences = new ArrayList<List<String>>();
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), INPUT_ENC));
			int sentenceCount = 0;
			while ((maxSentences < 0 || sentenceCount < maxSentences) && in.ready()) {
				String line = in.readLine();
				List<String> sentence = readSentence(line);
				sentences.add(sentence);
				sentenceCount++;
			}
//			System.out.printf("Read %d sentences.%n", sentences.size());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return sentences;
	}

	private static List<String> readSentence(String line) {
		List<String> words = new ArrayList<String>();
		String[] tokens = line.split("\\s+");
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (token.equals("<s")) continue;
			if (token.equals("</s>")) continue;
			if (token.startsWith("snum=")) {
				continue;
			}
			words.add(Interners.stringInterner.intern(token));
		}
		return words;
	}

}
