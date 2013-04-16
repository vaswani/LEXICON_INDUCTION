package translex;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import fig.basic.MapUtils;

public class BilingualLexicon {
	public Map<String, Set<String>> map = new HashMap<String, Set<String>>();
	Set<String> allTranslations = new HashSet<String>();
	
	public BilingualLexicon() {
		
	}
	
	public boolean containsTranslation(String dom, String codom) {
		Set<String> translations = map.get(dom);
		if (translations == null) { return false; }
		return translations.contains(codom);
	}
	
	public boolean containsTranslation(String dom) {
		Set<String> translations = map.get(dom);
		return translations != null && !translations.isEmpty();
	}
	
	public void addTranslation(String dom, String codom) {
		Set<String> translations = MapUtils.getMut(map, dom, new HashSet<String>());
		translations.add(codom);
		allTranslations.add(codom);
	}
	
	public Set<String> getAllImage() {
		return allTranslations;
	}
	
	public int size() {
		return map.size();
	}
	
	public Set<String> getImage(String domWord) {
		return MapUtils.get(map, domWord, new HashSet<String>());
	}

	
}
