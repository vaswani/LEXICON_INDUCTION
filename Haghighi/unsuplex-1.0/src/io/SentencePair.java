package io;

import java.util.List;

/**
 * A holder for a pair of sentences, each a list of strings.  Sentences in
 * the test sets have integer IDs, as well, which are used to retreive the
 * gold standard alignments for those sentences.
 */
public class SentencePair {
	int sentenceID;
	String sourceFile;
	List<String> domainWords;
	List<String> codomainWords;

//	public int getSentenceID() {
//		return sentenceID;
//	}

//	public String getSourceFile() {
//		return sourceFile;
//	}

	public List<String> getDomainWords() {
		return domainWords;
	}

	public List<String> getCodomainWords() {
		return codomainWords;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int domainPosition = 0; domainPosition < domainWords.size(); domainPosition++) {
			String domainWord = domainWords.get(domainPosition);
			sb.append(domainPosition);
			sb.append(":");
			sb.append(domainWord);
			sb.append(" ");
		}
		sb.append("\n");
		for (int codomainPosition = 0; codomainPosition < codomainWords.size(); codomainPosition++) {
			String codomainWord = codomainWords.get(codomainPosition);
			sb.append(codomainPosition);
			sb.append(":");
			sb.append(codomainWord);
			sb.append(" ");
		}
		sb.append("\n");
		return sb.toString();
	}

	public SentencePair(int sentenceID, String sourceFile, List<String> domainWords, List<String> codomainWords) {
		this.sentenceID = sentenceID;
		this.sourceFile = sourceFile;
		this.domainWords = domainWords;
		this.codomainWords = codomainWords;
	}
	
}