package io;

import java.util.HashSet;
import java.util.Set;

import fig.basic.Pair;

/**
 * Alignments serve two purposes, both to indicate your system's guessed
 * alignment, and to hold the gold standard alignments.  Alignments map index
 * pairs to one of three values, unaligned, possibly aligned, and surely
 * aligned.  Your alignment guesses should only contain sure and unaligned
 * pairs, but the gold alignments contain possible pairs as well.
 *
 * To build an alignemnt, start with an empty one and use
 * addAlignment(i,j,true).  To display one, use the render method.
 */
public class Alignment {
	Set<Pair<Integer, Integer>> sureAlignments;
	Set<Pair<Integer, Integer>> possibleAlignments;

	public boolean containsSureAlignment(int domainPosition, int codomainPosition) {
		return sureAlignments.contains(new Pair<Integer, Integer>(domainPosition, codomainPosition));
	}

	public boolean containsPossibleAlignment(int domainPosition, int codomainPosition) {
		return possibleAlignments.contains(new Pair<Integer, Integer>(domainPosition, codomainPosition));
	}

	public void addAlignment(int domainPosition, int codomainPosition, boolean sure) {
		Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(domainPosition, codomainPosition);
		if (sure)
			sureAlignments.add(alignment);
		possibleAlignments.add(alignment);
	}

	public Alignment() {
		sureAlignments = new HashSet<Pair<Integer, Integer>>();
		possibleAlignments = new HashSet<Pair<Integer, Integer>>();
	}
	
	public static String render(Alignment alignment, SentencePair sentencePair) {
		return render(alignment, alignment, sentencePair);
	}

	public static String render(Alignment reference, Alignment proposed, SentencePair sentencePair) {
		StringBuilder sb = new StringBuilder();
		for (int codomainPosition = 0; codomainPosition < sentencePair.getCodomainWords().size(); codomainPosition++) {
			for (int domainPosition = 0; domainPosition < sentencePair.getDomainWords().size(); domainPosition++) {
				boolean sure = reference.containsSureAlignment(domainPosition, codomainPosition);
				boolean possible = reference.containsPossibleAlignment(domainPosition, codomainPosition);
				char proposedChar = ' ';
				if (proposed.containsSureAlignment(domainPosition, codomainPosition))
					proposedChar = '#';
				if (sure) {
					sb.append('[');
					sb.append(proposedChar);
					sb.append(']');
				} else {
					if (possible) {
						sb.append('(');
						sb.append(proposedChar);
						sb.append(')');
					} else {
						sb.append(' ');
						sb.append(proposedChar);
						sb.append(' ');
					}
				}
			}
			sb.append("| ");
			sb.append(sentencePair.getCodomainWords().get(codomainPosition));
			sb.append('\n');
		}
		for (int domainPosition = 0; domainPosition < sentencePair.getDomainWords().size(); domainPosition++) {
			sb.append("---");
		}
		sb.append("'\n");
		boolean printed = true;
		int index = 0;
		while (printed) {
			printed = false;
			StringBuilder lineSB = new StringBuilder();
			for (int domainPosition = 0; domainPosition < sentencePair.getDomainWords().size(); domainPosition++) {
				String domainWord = sentencePair.getDomainWords().get(domainPosition);
				if (domainWord.length() > index) {
					printed = true;
					lineSB.append(' ');
					lineSB.append(domainWord.charAt(index));
					lineSB.append(' ');
				} else {
					lineSB.append("   ");
				}
			}
			index += 1;
			if (printed) {
				sb.append(lineSB);
				sb.append('\n');
			}
		}
		return sb.toString();
	}
}