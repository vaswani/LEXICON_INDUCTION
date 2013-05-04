package translex;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.Map.Entry;


import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Pair;

public class ProbabilisticTranslationLexicon extends TranslationLexicon {
	
	private Set<String> domain;
	private Set<String> codomain;
	private Set<String> preimage;
	private Set<String> image;
	private CounterMap<String,String> map;
	private CounterMap<String,String> inverseMap;
	
	public ProbabilisticTranslationLexicon(Collection<String> domain0, Collection<String> codomain0) {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new CounterMap<String,String>();
		this.inverseMap = new CounterMap<String,String>();
		
		for (String word : domain0) {
			domain.add(word);
		}
		for (String word : codomain0) {
			codomain.add(word);
		}
	}
	
	public ProbabilisticTranslationLexicon(ProbabilisticTranslationLexicon transLex) {
		this(transLex.domain, transLex.codomain, transLex.map);
	}
	
	public ProbabilisticTranslationLexicon(Collection<String> domain, Collection<String> codomain, ProbabilisticTranslationLexicon transLex) {
		this(domain, codomain, transLex.map);
	}
	
	public ProbabilisticTranslationLexicon(Collection<String> domain0, Collection<String> codomain0, CounterMap<String,String> map0) {
		this.domain = new HashSet<String>();
		this.codomain = new HashSet<String>();
		this.preimage = new HashSet<String>();
		this.image = new HashSet<String>();
		this.map = new CounterMap<String,String>();
		this.inverseMap = new CounterMap<String,String>();

		for (String word : domain0) {
			domain.add(word);
		}
		for (String word : codomain0) {
			codomain.add(word);
		}
		for (String domainWord : domain) {
			if (map0 != null && map0.containsKey(domainWord)) {
				for (Entry<String,Double> entry : map0.getCounter(domainWord).getEntrySet()) {
					String codomainWord = entry.getKey();
					double value = entry.getValue();
					if (codomain.contains(codomainWord) && value > 0) {
						map.setCount(domainWord, codomainWord, value);
						preimage.add(domainWord);
						inverseMap.setCount(codomainWord, domainWord, value);
						image.add(codomainWord);
					}
				}
			}
		}
	}
	
	public boolean isFunction() {
		for (String word : preimage) {
			if (map.getCounter(word).size() != 1)
				return false;
		}
		return true;
	}

	public boolean isOneToOne() {
		return isFunction() && preimage.size() == image.size();
	}
	
	public void addTranslation(String domainWord, String codomainWord) {
		domain.add(domainWord);
		preimage.add(domainWord);
		codomain.add(codomainWord);
		image.add(codomainWord);
		map.setCount(domainWord, codomainWord, 1.0);
		inverseMap.setCount(codomainWord, domainWord, 1.0);
	}
	
	public void addTranslation(String domainWord, String codomainWord, double score) {
		if (!(score > 0))
			throw new IllegalArgumentException("Values of translation lexicon mappings must be positive.");
		domain.add(domainWord);
		preimage.add(domainWord);
		codomain.add(codomainWord);
		image.add(codomainWord);
		map.setCount(domainWord, codomainWord, score);
		inverseMap.setCount(codomainWord, domainWord, score);
	}
	
	public void removeTranslation(String domainWord, String codomainWord) {
		throw new UnsupportedOperationException();
	}
	
	public boolean containsTranslation(String domainWord, String codomainWord) {
		if (map.getCount(domainWord, codomainWord) > 0.0)
			return true;
		return false;
	}
	
	public double getTranslationValue(String domainWord, String codomainWord) {
		return map.getCount(domainWord, codomainWord);
	}
	
	public String getBestTranslationToCodomain(String domainWord) {
		return map.getCounter(domainWord).argMax();
	}
	
	public String getBestTranslationToDomain(String codomainWord) {
		return inverseMap.getCounter(codomainWord).argMax();
	}
	
	public Counter<String> getImageCounter(String domainWord) {
		return map.getCounter(domainWord);
	}
	
	public Counter<String> getPreimageCounter(String codomainWord) {
		return inverseMap.getCounter(codomainWord);
	}
	
	public boolean isEmpty() {
		return map.isEmpty();
	}
	
	public Set<String> getDomain() {
		return domain;
	}
	
	public Set<String> getCodomain() {
		return codomain;
	}
	
	public Set<String> getPreimage() {
		return preimage;
	}
	
	public Set<String> getImage() {
		return image;
	}

	public Set<String> getImage(String domainWord) {
		return map.getCounter(domainWord).keySet();
	}

	public Set<String> getPreimage(String codomainWord) {
		return inverseMap.getCounter(codomainWord).keySet();
	}

	public Set<Pair<String, String>> getTranslationPairs() {
		Set<Pair<String,String>> transPairs = new HashSet<Pair<String,String>>();
	
		for (String domainWord : preimage) {
			for (String codomainWord : map.getCounter(domainWord).keySet()) {
				transPairs.add(Pair.newPair(domainWord, codomainWord));
			}
		}
		
		return transPairs;
	}

	public FunctionalTranslationLexicon asFunctionalTranslationLexicon(Collection<String> domain0, Collection<String> codomain0) {
		FunctionalTranslationLexicon funcTransLex = new FunctionalTranslationLexicon(domain0, codomain0);
		
		for (String domainWord : domain0) {
			if (preimage.contains(domainWord)) {
				String codomainBest = map.getCounter(domainWord).argMax();
				if (codomain0.contains(codomainBest)) {
					funcTransLex.addMapping(domainWord, codomainBest);
				}
			}
		}
		
		return funcTransLex;
	}
	
}
