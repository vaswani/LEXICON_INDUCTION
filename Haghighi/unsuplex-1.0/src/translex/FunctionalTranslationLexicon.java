package translex;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import fig.basic.Pair;


public class FunctionalTranslationLexicon extends TranslationLexicon implements Map<String,String> {
	
	private Set<String> domain;
	private Set<String> codomain;
	private Set<String> preimage;
	private Set<String> image;
	private Map<String,String> map;
	private Map<String,Set<String>> inverseMap;
	
	public FunctionalTranslationLexicon() {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,String>();
		this.inverseMap = new HashMap<String,Set<String>>();
	}
	
	public FunctionalTranslationLexicon(Collection<String> domain0, Collection<String> codomain0) {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,String>();
		this.inverseMap = new HashMap<String,Set<String>>();
		for (String word : domain0) {
			domain.add(word);
		}
		for (String word : codomain0) {
			codomain.add(word);
		}
	}
	
	public FunctionalTranslationLexicon(Collection<String> domain0, Collection<String> codomain0, TranslationLexicon transLex0) {
		if (!transLex0.isFunction()) {
			throw new IllegalArgumentException("Cannot build FunctionalTranslationLexicon from non-functional TranslationLexicon.");
		}
		
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,String>();
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
	
	// TranslationLexicon implementation
	
	public void addTranslation(String domainWord, String codomainWord) {
		if (preimage.contains(domainWord)) {
			throw new IllegalStateException("Word already mapped.");
		}
		
		domain.add(domainWord);
		preimage.add(domainWord);
		codomain.add(codomainWord);
		image.add(codomainWord);
		map.put(domainWord,codomainWord);
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
		map.remove(domainWord);
		preimage.remove(domainWord);
		Set<String> wordPreimage = inverseMap.get(codomainWord);
		if (wordPreimage != null && wordPreimage.contains(domainWord)) {
			wordPreimage.remove(domainWord);
			if (wordPreimage.isEmpty()) {
				inverseMap.remove(codomainWord);
				image.remove(codomainWord);
			}
		}
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
		return inverseMap.get(codomainWord);
	}

	public boolean isEmpty() {
		return map.isEmpty();
	}

	public boolean isFunction() {
		return true;
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
		for (String domainWord : preimage) {
			transPairs.add(Pair.newPair(domainWord, map.get(domainWord)));
		}
		return transPairs;
	}

	// (Graph package) Mapping interface
	
	public String getMapping(String key) {
		for (String value : getImage(key))
			return value;
		return null;
	}
	
	public String getInverseMapping(String value) {
		for (String key : getPreimage(value))
			return key;
		return null;
	}
	
	public void addMapping(String key, String value) {
		addTranslation(key, value);
	}

	public void removeMapping(String key) {
		if (getPreimage().contains(key)) {
			String value = getMapping(key);
			removeTranslation(key,value);
		}
	}
	
	public boolean containsMapping(String key, String value) {
		return containsTranslation(key, value);
	}

	public Set<Pair<String, String>> getMappingPairs() {
		return getTranslationPairs();
	}
	
	// Map interface

	public void clear() {
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new HashMap<String,String>();
		this.inverseMap = new HashMap<String,Set<String>>();
	}

	public boolean containsKey(Object key) {
		return map.containsKey(key);
	}

	public boolean containsValue(Object val) {
		return map.containsKey(val);
	}

	public Set<java.util.Map.Entry<String, String>> entrySet() {
		return map.entrySet();
	}

	public String get(Object key) {
		return map.get(key);
	}

	public Set<String> keySet() {
		return map.keySet();
	}

	public String put(String key, String value) {
		String result = get(key);
		addTranslation(key, value);
		return result;
	}

	public void putAll(Map<? extends String, ? extends String> map0) {
		for (java.util.Map.Entry<? extends String, ? extends String> entry : map0.entrySet()) {
			put(entry.getKey(), entry.getValue());
		}	
	}

	public String remove(Object key) {
		String result = get(key);
		removeTranslation((String) key, result);
		return result;
	}

	public int size() {
		return map.entrySet().size();
	}

	public Collection<String> values() {
		return map.values();
	}

}
