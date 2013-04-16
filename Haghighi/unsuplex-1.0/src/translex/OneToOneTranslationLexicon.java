package translex;

import java.util.Collection;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import fig.basic.Pair;

public class OneToOneTranslationLexicon extends TranslationLexicon {
	
	private Set<String> domain;
	private Set<String> codomain;
	private Set<String> preimage;
	private Set<String> image;
	private Map<String,String> map;
	private Map<String,String> inverseMap;
	
	public OneToOneTranslationLexicon(Collection<String> domain0, Collection<String> codomain0) {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,String>();
		this.inverseMap = new HashMap<String,String>();
		for (String word : domain0) {
			domain.add(word);
		}
		for (String word : codomain0) {
			codomain.add(word);
		}
	}
	
	public OneToOneTranslationLexicon(Collection<String> domain0, Collection<String> codomain0, TranslationLexicon transLex0) {
		if (!transLex0.isOneToOne()) {
			throw new IllegalArgumentException("Cannot build OneToOneTranslationLexicon from non-one-to-one TranslationLexicon.");
		}
		
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,String>();
		this.inverseMap = new HashMap<String,String>();
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
		if (preimage.contains(domainWord) || image.contains(codomainWord)) {
			throw new IllegalStateException("Preimage or image already covered.");
		}
		
		domain.add(domainWord);
		preimage.add(domainWord);
		codomain.add(codomainWord);
		image.add(codomainWord);
		map.put(domainWord, codomainWord);
		inverseMap.put(codomainWord, domainWord);
	}
	
	public void removeTranslation(String domainWord, String codomainWord) {
		preimage.remove(domainWord);
		image.remove(codomainWord);
		map.remove(domainWord);
		inverseMap.remove(codomainWord);
	}

	public boolean containsTranslation(String domainWord, String codomainWord) {
		if (preimage.contains(domainWord)) {
			return map.get(domainWord) == codomainWord;
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
		Set<String> result = new HashSet<String>();
		result.add(map.get(domainWord));
		return result;
	}
	
	public Set<String> getPreimage(String codomainWord) {
		Set<String> result = new HashSet<String>();
		result.add(inverseMap.get(codomainWord));
		return result;
	}
	
	public boolean isEmpty() {
		return map.isEmpty();
	}

	public boolean isFunction() {
		return true;
	}

	public boolean isOneToOne() {
		return true;
	}

	public Set<Pair<String, String>> getTranslationPairs() {
		Set<Pair<String,String>> transPairs = new HashSet<Pair<String,String>>();
		for (Map.Entry<String,String> entry : map.entrySet()) {
			transPairs.add(Pair.newPair(entry.getKey(), entry.getValue()));
		}
		return transPairs;
	}

}
