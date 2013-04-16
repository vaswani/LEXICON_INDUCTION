package inducer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import translex.BilingualLexicon;

import edu.berkeley.nlp.util.BoundedList;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.ConcatenationIterator;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.PriorityQueue;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.MapUtils;
import fig.basic.Option;
import fig.basic.Pair;

import io.BitextCorpusExtractor;
import io.POSTagPrefixes;
import io.POSTagPrefixes.POSTag;

public class NewBitext {
	
	BitextCorpusExtractor bitextCorpusExtractor;
	String basePath ;
	public static Options opts = new Options();
	
	public static class Options {
		@Option
		public String domainCorpusPath = "source.corpus";
		@Option
		public String codomainCorpusPath = "target.corpus";
		@Option
		public String domainAdditionalCorpusPath = "source-additional.corpus.gz";
		@Option
		public String codomainAdditionalCorpusPath = "target-additional.corpus.gz";
		@Option
		public String domainPOSPath = "source.tagmap";
		@Option
		public String codomainPOSPath = "target.tagmap";
		@Option
		public String lexiconPath = "lexicon";
	}
	
	Counter<String> domainCounts ;
	Counter<String> codomainCounts ;
	List<List<String>> domainCorpus ;
	List<List<String>> codomainCorpus ;
	Map<String, Set<POSTag>> domainPOSMap ;
	Map<String, Set<POSTag>> codomainPOSMap ;
	BilingualLexicon bilingualLex ;
	int maxAdditionalSents ;
	
	public static interface SentenceCallbackFunction {
		public void callback(List<String> sent);
	}
	
	private void doSentenceCallback(String path, int maxSentences, SentenceCallbackFunction fn) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				String[] words = line.split("\\s+");
				fn.callback(Arrays.asList(words));
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	private List<List<String>> readCorpus(String path, int maxSentences) {
		List<List<String>> corpus = new ArrayList<List<String>>();
		try {
			FileInputStream fi = new FileInputStream(path);
			Reader ir = new InputStreamReader(fi, "UTF8"); // hardcoded
			BufferedReader br = new BufferedReader(ir);
			while (corpus.size() < maxSentences) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				List<String> words = CollectionUtils.makeList(line.split("\\s+"));
//				for (int i=0; i < words.size(); ++i) {
//					String word = words.get(i).toLowerCase();
//					words.set(i, word)
//				}
				corpus.add(words);
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
		return corpus;
	}
	
	private Map<String, Set<POSTag>> readTagMap(String path, boolean dom) {
		
		Map<String, Set<POSTag>> posMap = new HashMap<String, Set<POSTag>>();
		try {
			FileInputStream fi = new FileInputStream(path);
			Reader ir = new InputStreamReader(fi, "UTF8"); // hardcoded
			BufferedReader br = new BufferedReader(ir);
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				String[] fields = line.split("\\s+");
				String word = fields[0];
				if ((dom && seenDomainWord(word)) || (!dom && seenCodomainWord(word))) {
					Set<POSTag> tags = MapUtils.getMut(posMap, word, new HashSet<POSTag>());
					for (int i=1; i < fields.length; ++i) {
						POSTag tag = POSTagPrefixes.getPOSTag(fields[i],dom);
						tags.add(tag);
					}					
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}		
		return posMap;

	}
	
	public Pair<Counter<String>, Counter<String>> getWordCounts() {
		return Pair.newPair(domainCounts, codomainCounts);
	}
	
	private Counter<String> getWordCounts(List<List<String>> corpora) {
		Counter<String> counts = new Counter<String>();
		for (List<String> sent: corpora) {
			counts.incrementAll(sent, 1.0);
		}
		return counts;
	}
	
	public NewBitext(String basePath, BitextCorpusExtractor corpusExtractor, int maxSentences, int maxAdditionalSents) {
		this.bitextCorpusExtractor = corpusExtractor == null ? new BitextCorpusExtractor.NoOpExtractor(): corpusExtractor;
		this.basePath = basePath;
		
		List<List<String>> rawDomainCorpus = readCorpus(basePath + "/" + opts.domainCorpusPath, maxSentences);
		List<List<String>> rawCodomainCorpus = readCorpus(basePath + "/" + opts.codomainCorpusPath, maxSentences);
		Pair<List<List<String>>,List<List<String>>> pair = bitextCorpusExtractor.extractCorpus(Pair.newPair(rawDomainCorpus, rawCodomainCorpus));
		domainCorpus = pair.getFirst();
		domainCounts = getWordCounts(domainCorpus);
		
		codomainCorpus = pair.getSecond();
		codomainCounts = getWordCounts(codomainCorpus);
		
		LogInfo.logs("Bitext Corpus Extractor: %s\n", this.bitextCorpusExtractor);
		LogInfo.logs("Domain Corpus Size: %d sentences\n", domainCorpus.size());
		LogInfo.logs("Codomain Corpus Size: %d sentences\n", codomainCorpus.size());
		
		
		domainPOSMap = readTagMap(basePath + "/" + opts.domainPOSPath,true);		
		codomainPOSMap = readTagMap(basePath + "/" + opts.codomainPOSPath,false);
		
		bilingualLex = readLexicon(basePath + "/" + opts.lexiconPath);
		LogInfo.logs("Size of Lexicon: %d entries\n", bilingualLex.size());
		
		this.maxAdditionalSents = maxAdditionalSents;
	}
	
	private boolean seenDomainWord(String word) {
		 return domainCounts.containsKey(word);		
	}
	
	private boolean seenCodomainWord(String word) {
		return codomainCounts.containsKey(word);
	}

	private BilingualLexicon readLexicon(String path) {
		BilingualLexicon bilingualLex = new BilingualLexicon();
		try {
			FileInputStream fi = new FileInputStream(path);
			Reader ir = new InputStreamReader(fi, "UTF8"); // hardcoded
			BufferedReader br = new BufferedReader(ir);
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				try {
				String[] fields = line.split("\\s+");
//				assert fields.length == 2;
				String domWord = fields[0];
				String codomWord = fields[1];
				if (seenDomainWord(domWord) && seenCodomainWord(codomWord)) {
					bilingualLex.addTranslation(domWord, codomWord);	
				} 
				} catch (Exception e) { }
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
		LogInfo.logs("read %d entries in lexicon",bilingualLex.size());
		return bilingualLex;
	}
	
	private Indexer<String> getMostCommonWordsInOrder(Counter<String> counts, Map<String, Set<POSTag>> tagMap, int N, Set<POSTag> tags) {
		PriorityQueue<String> pq = counts.asPriorityQueue();
		Indexer<String> topWords = new Indexer<String>();
		
		while (topWords.size() < N && pq.hasNext()) {
			String word = pq.next();
			Set<POSTag> wordTags = tagMap.get(word);
			if (wordTags == null) { continue; }
			boolean pass = false;
			for (POSTag t: tags) {
				if (wordTags.contains(t)) {
					pass = true; break;
				}
			}
			if (pass) {
				topWords.add(word);
			}
		}
		
		return topWords;
	}
	
	public Pair<Indexer<String>, Indexer<String>> getMostCommonWordsInOrder(int N, Set<POSTag> tags) {
		Indexer<String> topDomain = getMostCommonWordsInOrder(domainCounts, domainPOSMap, N, tags);
		Indexer<String> topCodomain = getMostCommonWordsInOrder(codomainCounts, codomainPOSMap, N, tags);
		return Pair.newPair(topDomain, topCodomain);
	}
	
	private static List<String> filter(List<String> sent, Map<String, Set<POSTag>> tagMap, Set<POSTag> tags) {
		List<String> filtered = new ArrayList<String>();
		for (String tok: sent) {
			Set<POSTag> tokTags = tagMap.get(tok);
			if (tokTags == null) {
				continue;
			}
			boolean foundMatch = false;
			for (POSTag t: tags) { 
				if (tokTags.contains(t)) {
					foundMatch = true;
					break;
				}				
			}
			if (foundMatch) {
				filtered.add(tok);
			}
		}
		return filtered;
	}
	
	private List<List<String>> reduceCorpus(List<List<String>> corpus, Map<String, Set<POSTag>> tagMap, Set<POSTag> tags) {
		List<List<String>> reducedCorpus = new ArrayList<List<String>>();
		int numToks = 0;
		for (List<String> sent: corpus) {
			List<String> filteredSent = filter(sent, tagMap, tags);
			reducedCorpus.add(filteredSent);
			numToks += filteredSent.size();
		}
		LogInfo.logs("Filtering on %s we have %d toks",tags.toString(),numToks);
		return reducedCorpus;
	}
			
	
	public Pair<List<List<String>>,List<List<String>>> getReducedBaseCorpus(Set<POSTag> tags) {
		List<List<String>> reducedDomain = reduceCorpus(domainCorpus, domainPOSMap, tags);
		List<List<String>> reducedCodomain = reduceCorpus(codomainCorpus, codomainPOSMap, tags);
		return Pair.newPair(reducedDomain, reducedCodomain);
	}
	
	public class SentenceIterator implements Iterator<List<String>> {

		boolean noTagFilter = false;
		BufferedReader reader ;
		String nextLine ;
		Set<POSTag> tags;
		Map<String, Set<POSTag>> tagMap ;
		int count = 0;
		int max = 0;
				
		public SentenceIterator(String path, Map<String, Set<POSTag>> tagMap, Set<POSTag> tags, int max) {
			this.tags = tags;
			this.tagMap = tagMap;
			this.max = max;
			noTagFilter = (tags == null) || tags.contains(POSTag.UNDEFINED);
			try {
				reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(path))));
				buffer();
			} catch (Exception e) { 
				e.printStackTrace();
				System.exit(0);
			}
		}
		
		private void buffer() {
			try {
				nextLine = reader.readLine();
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
				
		public boolean hasNext() {
			return nextLine != null && count < max;
		}

		public List<String> next() {
			List<String> words = Arrays.asList(nextLine.split("\\s+"));
			buffer();  // get next line
			if (noTagFilter) { return words; }
			List<String> filteredWords = filter(words, tagMap, tags);
			return filteredWords;
		}

		public void remove() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException();
		}
		
		
	}
	
	private static class Beam<T> {
		List<T> items = new ArrayList<T>();
		List<Double> scores = new ArrayList<Double>();
		int maxItems ;
		
		public Beam(int maxItems) {
			this.maxItems = maxItems;			
		}
		
		@Override
		public String toString() {
			// TODO Auto-generated method stub
			List<Pair<T,Double>> itemsWithScores = new ArrayList<Pair<T,Double>>();
			for (int i=0; i < items.size(); ++i) {
				itemsWithScores.add(Pair.newPair(items.get(i), scores.get(i)));
			}
			return itemsWithScores.toString();
		}
		
		public void add(T item, double score) {
			
			if (items.size() < maxItems) {
				int index = findIndex(score);
				insert(index, item, score);
				return;
			}
			assert items.size() == maxItems;
			if (score <= scores.get(maxItems-1)) {
				return;
			}
			int index = findIndex(score);
			assert index <= maxItems-1 : "invalid index" + index;
			insert(index,item,score);
			if (items.size() > maxItems) {
				items.remove(maxItems);
				scores.remove(maxItems);
			} else {
				assert false;
			}
		}
		
		private void insert(int index, T item, double score) {
			items.add(index, item);
			scores.add(index, score);
		}
		
//		private boolean check() {
//			for (int i=0; i+1 < scores.size(); ++i) {
//				if (scores.get(i) < scores.get(i+1)) {
//					return false;
//				}
//			}
//			return true;
//		}
		
		private int findIndex(double score) {
//			int index = Collections.binarySearch(scores, score, new Comparator<Double>() {
//
//				public int compare(Double o1, Double o2) {
//					return (int) (o2-o1);
//				}
//
//				
//				
//			});
//			index = index >= 0 ? index : (-index-1);
//			return index;
			if (scores.size() > maxItems && score < scores.get(scores.size()-1)) {
				return scores.size();
			}
			int i=0;
			for (i=0; i < scores.size(); ++i) {
				if (score > scores.get(i)) {
					return i;
				}
			}
			return i;
		}
	}
	
	public Pair<? extends Iterator<List<String>>,? extends Iterator<List<String>>> getReducedFullCorpus(Set<POSTag> tags, Counter<String> domWordScores, Counter<String> codomWordScores) {
		Collection<Iterator<List<String>>> domIters = new ArrayList<Iterator<List<String>>>();
		Collection<Iterator<List<String>>> codomIters = new ArrayList<Iterator<List<String>>>();
				
		List<List<String>> reducedDomain = reduceCorpus(domainCorpus, domainPOSMap, tags);
		domIters.add(reducedDomain.iterator());
		List<List<String>> reducedCodomain = reduceCorpus(codomainCorpus, codomainPOSMap, tags);
		codomIters.add(reducedCodomain.iterator());
		
		 
		Beam<List<String>> domBeam = new Beam<List<String>>(maxAdditionalSents);
		// Additional Domain
		String domAdditionalPath = basePath + "/" + opts.domainAdditionalCorpusPath; 
		if (exists(domAdditionalPath)) {
			LogInfo.track("Reading at most %d additional dom sentences from %s",maxAdditionalSents,domAdditionalPath);
			SentenceIterator domAdditionalIt = new SentenceIterator(domAdditionalPath,domainPOSMap,tags, 100*maxAdditionalSents);
			while (domAdditionalIt.hasNext()) {
				List<String> sent = domAdditionalIt.next();
				double score = 0.0;
				for (String word: sent) {
					score += domWordScores.getCount(word);
				}
				domBeam.add(sent, score);
			}
			domIters.add(domBeam.items.iterator());
			LogInfo.end_track();
		}
		// Additional Codomain
		String codomAdditionalPath = basePath + "/" + opts.codomainAdditionalCorpusPath;
		Beam<List<String>> codomBeam = new Beam<List<String>>(maxAdditionalSents);
		if (exists(codomAdditionalPath)) {
			LogInfo.track("Reading at most %d additional codom sentences from %s",maxAdditionalSents, codomAdditionalPath);			
			SentenceIterator codomAdditionalIt = new SentenceIterator(codomAdditionalPath,codomainPOSMap,tags,100*maxAdditionalSents);
			while (codomAdditionalIt.hasNext()) {
				List<String> sent =  codomAdditionalIt.next();
				double score = 0.0;
				for (String word: sent) {
					score += codomWordScores.getCount(word);
				}
				codomBeam.add(sent, score);
			}
			codomIters.add(codomBeam.items.iterator());
			LogInfo.end_track();
		}
						
		ConcatenationIterator<List<String>> domConcatIter = new ConcatenationIterator<List<String>>(domIters);
		ConcatenationIterator<List<String>> codomConcatIter = new ConcatenationIterator<List<String>>(codomIters);
		return Pair.newPair(domConcatIter, codomConcatIter);
	}
	
	private boolean exists(String path) {
		return new File(path).exists();
	}

	public BilingualLexicon getLexicon() {
		return bilingualLex;
	}

	public Pair<List<List<String>>, List<List<String>>> getBaseCorpus() {
		return Pair.newPair(domainCorpus, codomainCorpus);
	}
	
	
	
}
