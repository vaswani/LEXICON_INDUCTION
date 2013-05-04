package io;

import java.util.HashSet;
import java.util.Set;

public class POSTagPrefixes {

	public static final String DOMAIN_NOUN_PREFIX = "N";
	public static final String DOMAIN_VERB_PREFIX = "V";
	public static final String DOMAIN_DETERMINER_PREFIX = "D";
	public static final String CODOMAIN_NOUN_PREFIX = "N";
	public static final String CODOMAIN_VERB_PREFIX = "V";
	public static final String CODOMAIN_DETERMINER_PREFIX = "D";

	public enum POSTag {
		VERB, NOUN, DETERMINER, UNDEFINED
	}
	
	public static Set<POSTagPrefixes.POSTag> getTagSet(String tagOpts) {
		Set<POSTagPrefixes.POSTag> tagSet = new HashSet<POSTagPrefixes.POSTag>();
		for (POSTagPrefixes.POSTag tag : POSTagPrefixes.POSTag.values()) {
			if (tagOpts.contains("all") || tagOpts.contains(tag.toString().toLowerCase())) {
				tagSet.add(tag);
			}
		}
		return tagSet;
	}
	
	private static boolean startsWithIgnoreCase(String a, String b) {
		return a.toLowerCase().startsWith(b.toLowerCase());
	}

	// doesn't work for UNDEFINED
	public static boolean corpusTagMatch(String corpusTag, POSTag tag, boolean dom) {
		boolean result = false;
		switch (tag) {
		case VERB:
			result = (dom) ? startsWithIgnoreCase(corpusTag, DOMAIN_VERB_PREFIX) : startsWithIgnoreCase(corpusTag, CODOMAIN_VERB_PREFIX);
			break;
		case NOUN:
			result = (dom) ? startsWithIgnoreCase(corpusTag, DOMAIN_NOUN_PREFIX) : startsWithIgnoreCase(corpusTag, CODOMAIN_NOUN_PREFIX);
			break;
		case DETERMINER:
			result = (dom) ? startsWithIgnoreCase(corpusTag, DOMAIN_DETERMINER_PREFIX) : startsWithIgnoreCase(corpusTag, CODOMAIN_DETERMINER_PREFIX);
			break;
		}
		return result;
	}
	
	public static POSTag getPOSTag(String corpusTag, boolean dom) {
		for (POSTag tag : POSTag.values()) {
			if (corpusTagMatch(corpusTag, tag, dom)) {
				return tag;
			}
		}
		return POSTag.UNDEFINED;
	}


}
