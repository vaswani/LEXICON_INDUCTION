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

import fig.basic.*;

public class BitextIO extends TextIO {
	
	public static List<Alignment> readAlignments(String baseFileName, String alignExtension, int maxSentences) {
		String fileName = baseFileName + "." + alignExtension;
		List<Alignment> alignments = new ArrayList<Alignment>();
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), INPUT_ENC));
			int sentenceCount = 0;
			while ((maxSentences < 0 || sentenceCount < maxSentences) && in.ready()) {
				Alignment alignment = new Alignment();
				String line = in.readLine();
				String[] sentenceAlign = line.split("\\s+");
				for (String positionString : sentenceAlign) {
					String[] positions = positionString.split("-");
					Integer codomainPosition = Integer.parseInt(positions[0]);
					Integer domainPosition = Integer.parseInt(positions[1]);
					alignment.addAlignment(domainPosition, codomainPosition, true);
				}
				alignments.add(alignment);
				sentenceCount++;
			}
//			System.out.printf("Read %d alignments.%n", alignments.size());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return alignments;
	}
	
	public static void writeSentencePairs(List<SentencePair> sentencePairs, String baseFileName, String domainExtension, String codomainExtension) {
		String domainFileName = baseFileName + "." + domainExtension;
		String codomainFileName = baseFileName + "." + codomainExtension;
		
		try {
			BufferedWriter domainOut = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(domainFileName), OUTPUT_ENC));
			BufferedWriter codomainOut = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(codomainFileName), OUTPUT_ENC));
			for (SentencePair sentPair : sentencePairs) {
				writeSentence(domainOut, sentPair.getDomainWords());
				domainOut.newLine();
				domainOut.flush();
				writeSentence(codomainOut, sentPair.getCodomainWords());
				codomainOut.newLine();
				codomainOut.flush();
			}
			domainOut.close();
			codomainOut.close();
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
	
	public static List<SentencePair> readSentencePairs(String baseFileName, String domainExtension, String codomainExtension, int maxSentencePairs) {
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		String domainFileName = baseFileName + "." + domainExtension;
		String codomainFileName = baseFileName + "." + codomainExtension;
		try {
			BufferedReader domainIn = new BufferedReader(new InputStreamReader(new FileInputStream(domainFileName), INPUT_ENC));
			BufferedReader codomainIn = new BufferedReader(new InputStreamReader(new FileInputStream(codomainFileName), INPUT_ENC));
			int sentenceCount = 0;
			while ((maxSentencePairs < 0 || sentenceCount < maxSentencePairs) && domainIn.ready() && codomainIn.ready()) {
				String domainLine = domainIn.readLine();
				String codomainLine = codomainIn.readLine();
				Pair<Integer,List<String>> domainSentenceAndID = readSentence(domainLine);
				Pair<Integer,List<String>> codomainSentenceAndID = readSentence(codomainLine);
				if (! domainSentenceAndID.getFirst().equals(codomainSentenceAndID.getFirst()))
					throw new RuntimeException("Sentence ID confusion in file "+baseFileName+", lines were:\n\t"+domainLine+"\n\t"+codomainLine);
				sentencePairs.add(new SentencePair(domainSentenceAndID.getFirst(), baseFileName, domainSentenceAndID.getSecond(), codomainSentenceAndID.getSecond()));
				sentenceCount++;
			}
//			System.out.printf("Read %d sentence pairs.%n", sentencePairs.size());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return sentencePairs;
	}

	private static Pair<Integer, List<String>> readSentence(String line) {
		int id = -1;
		List<String> words = new ArrayList<String>();
		String[] tokens = line.split("\\s+");
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (token.equals("<s")) continue;
			if (token.equals("</s>")) continue;
			if (token.startsWith("snum=")) {
				String idString = token.substring(5,token.length()-1);
				id = Integer.parseInt(idString);
				continue;
			}
			words.add(token.intern());
		}
		return new Pair<Integer, List<String>>(id, words);
	}

}
