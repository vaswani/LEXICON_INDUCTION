package translex;

import java.util.Collection;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import fig.basic.Pair;

public class GeneralTranslationLexicon extends TranslationLexicon {
	
	private Set<String> domain;
	private Set<String> codomain;
	private Set<String> preimage;
	private Set<String> image;
	private Map<String,Set<String>> map;
	private Map<String,Set<String>> inverseMap;
	
	public GeneralTranslationLexicon(Collection<String> domain0, Collection<String> codomain0) {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,Set<String>>();
		this.inverseMap = new HashMap<String,Set<String>>();
		for (String word : domain0) {
			domain.add(word);
		}
		for (String word : codomain0) {
			codomain.add(word);
		}
	}
	
	public GeneralTranslationLexicon(Collection<String> domain0, Collection<String> codomain0, TranslationLexicon transLex0) {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,Set<String>>();
		this.inverseMap = new HashMap<String,Set<String>>();
		for (String word : domain0) {
			domain.add(word);
		}
		for (String word : codomain0) {
			codomain.add(word);
		}
		for (Pair<String,String> transPair : transLex0.getTranslationPairs()) {
			String domainWord = transPair.getFirst();
			String codomainWord = transPair.getSecond();
			if (domain.contains(domainWord) && codomain.contains(codomainWord)) {
				addTranslation(domainWord, codomainWord);
			}	
		}
	}
		
	public void addTranslation(String domainWord, String codomainWord) {		
		domain.add(domainWord);
		preimage.add(domainWord);
		codomain.add(codomainWord);
		image.add(codomainWord);
		
		Set<String> wordImage = map.get(domainWord);
	 	if (wordImage != null) {
			wordImage.add(codomainWord);
		} else {
			wordImage = new HashSet<String>(); 
			wordImage.add(codomainWord);
			map.put(domainWord, wordImage);
		}
	 	
		Set<String> wordPreimage = inverseMap.get(codomainWord);
	 	if (wordPreimage != null) {
		    wordPreimage.add(domainWord);
		} else {
			wordPreimage = new HashSet<String>();
			wordPreimage.add(domainWord);
			inverseMap.put(codomainWord, wordPreimage);
		}
	}
	
	public void removeTranslation(String domainWord, String codomainWord) {
		Set<String> wordImage = map.get(domainWord);
		Set<String> wordPreimage = inverseMap.get(codomainWord);
		if (wordImage != null && wordImage.contains(codomainWord)) {
			wordImage.remove(codomainWord);
			if (wordImage.isEmpty()) {
				map.remove(domainWord);
				preimage.remove(domainWord);
			}
			wordPreimage.remove(domainWord);
			if (wordPreimage.isEmpty()) {
				inverseMap.remove(codomainWord);
				image.remove(codomainWord);
			}
		}
	}

	public boolean containsTranslation(String domainWord, String codomainWord) {
		if (preimage.contains(domainWord)) {
			return map.get(domainWord).contains(codomainWord);
		} else {
			return false;
		}
	}

	public Set<String> getCodomain() {
		return codomain;
	}

	public Set<String> getDomain() {
		return domain;
	}

	public Set<String> getImage() {
		return image;
	}

	public Set<String> getPreimage() {
		return preimage;
	}
	
	public Set<String> getImage(String domainWord) {
		return map.get(domainWord);
	}
	
	public Set<String> getPreimage(String codomainWord) {
		return inverseMap.get(codomainWord);
	}

	public boolean isEmpty() {
		return map.isEmpty();
	}

	public boolean isFunction() {
		boolean result = true;
		for (String domainWord : preimage) {
			result = result && (map.get(domainWord).size() == 1);
		}
		return result;
	}

	public boolean isOneToOne() {
		boolean result = true;
		for (String codomainWord : image) {
			result = result && (inverseMap.get(codomainWord).size() == 1);
		}
		return result;
	}

	public Set<Pair<String, String>> getTranslationPairs() {
		Set<Pair<String,String>> transPairs = new HashSet<Pair<String,String>>();
		for (Map.Entry<String,Set<String>> entry : map.entrySet()) {
			String domainWord = entry.getKey();
			for (String codomainWord : entry.getValue())
				transPairs.add(Pair.newPair(domainWord, codomainWord));
		}
		return transPairs;
	}

}
