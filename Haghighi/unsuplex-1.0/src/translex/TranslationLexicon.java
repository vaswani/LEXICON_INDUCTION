package translex;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import fig.basic.Pair;


public abstract class TranslationLexicon {
	
	public static String render(TranslationLexicon goldLexicon, TranslationLexicon seedLexicon, TranslationLexicon transLexicon) {
		StringBuffer buf = new StringBuffer();
		Set<Pair<String,String>> transPairs = transLexicon.getTranslationPairs();
		List<Pair<String,String>> sortedTransPairs = new ArrayList<Pair<String,String>>(transPairs);
		//Collections.sort(sortedTransPairs, new Pair.LexicographicPairComparator<String,String>(String.CASE_INSENSITIVE_ORDER, String.CASE_INSENSITIVE_ORDER));
		for (Pair<String,String> pair : sortedTransPairs) {
			String domainWord = pair.getFirst();
			String codomainWord = pair.getSecond();
			buf.append(pair.getFirst() + " : " + pair.getSecond() + "  " + (goldLexicon.containsTranslation(domainWord, codomainWord)?(seedLexicon.containsTranslation(domainWord, codomainWord)?"CORRECT":"NEW CORRECT"):"") + "\n");
		}
		return buf.toString();
	}
	
	public static String render(TranslationLexicon goldLexicon, TranslationLexicon transLexicon) {
		StringBuffer buf = new StringBuffer();
		Set<Pair<String,String>> transPairs = transLexicon.getTranslationPairs();
		List<Pair<String,String>> sortedTransPairs = new ArrayList<Pair<String,String>>(transPairs);
	//	Collections.sort(sortedTransPairs, new Pair.LexicographicPairComparator<String,String>(String.CASE_INSENSITIVE_ORDER, String.CASE_INSENSITIVE_ORDER));
		for (Pair<String,String> pair : sortedTransPairs) {
			String domainWord = pair.getFirst();
			String codomainWord = pair.getSecond();
			buf.append(pair.getFirst() + " : " + pair.getSecond() + "  " + (goldLexicon.containsTranslation(domainWord, codomainWord)?"CORRECT":"") + "\n");
		}
		return buf.toString();
	}
	
	
	public static String render(TranslationLexicon transLexicon) {
		StringBuffer buf = new StringBuffer();
		Set<Pair<String,String>> transPairs = transLexicon.getTranslationPairs();
		List<Pair<String,String>> sortedTransPairs = new ArrayList<Pair<String,String>>(transPairs);
//		Collections.sort(sortedTransPairs, new Pair.LexicographicPairComparator<String,String>(String.CASE_INSENSITIVE_ORDER, String.CASE_INSENSITIVE_ORDER));
		for (Pair<String,String> pair : sortedTransPairs) {
			buf.append(pair.getFirst() + " : " + pair.getSecond() +"\n");
		}
		return buf.toString();
	}
	
	public String toString() {
		return render(this);
	}
	
	public abstract boolean isFunction();

	public abstract boolean isOneToOne();
	
	public abstract void addTranslation(String domainWord, String codomainWord);
	
	public abstract void removeTranslation(String domainWord, String codomainWord);
	
	public abstract boolean containsTranslation(String domainWord, String codomainWord);
	
	public abstract Set<Pair<String,String>> getTranslationPairs();
	
	public abstract boolean isEmpty();
	
	public abstract Set<String> getDomain();
	
	public abstract Set<String> getCodomain();
	
	public abstract Set<String> getPreimage();

	public abstract Set<String> getImage();
	
	public abstract Set<String> getPreimage(String codomainWord);

	public abstract Set<String> getImage(String domainWord);
	
}
