package features;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.util.*;
import features.ContextKernelFeatureExtractor.ContextType;
import fig.basic.Indexer;
import inducer.DictionaryInducerTester.Options;
import inducer.LinearSparseKernel.KernelFeatureExtractor;
import inducer.LogUniqueContextCountInducer.UniqueContextCounter;
import io.Interners;

public class SimilarityKernelFeatureExtractor implements KernelFeatureExtractor<String> {

	private FeatureManager featManager;
	private CounterMap<String,Feature> sparseFeatureCounts;

	private static interface SameDomainSimilarityMeasure {
		public double getSimilarity(String word1, String word2);
	}
	
	private static class LogUniqueContextCountSimilarityMeasure implements SameDomainSimilarityMeasure {
		
		private UniqueContextCounter contextCounts;
		
		public LogUniqueContextCountSimilarityMeasure(List<List<String>> docs, Indexer<String> wordsToExtract, List<String> featWords, int windowSize, ContextType contextType) {
			this.contextCounts = new UniqueContextCounter(docs, wordsToExtract, windowSize, contextType);
		}

		public double getSimilarity(String word1, String word2) {
			double dist = Math.abs(Math.log(contextCounts.getUniqueContextCountRatio(word1)) - Math.log(contextCounts.getUniqueContextCountRatio(word2)));
			return -dist;
		}
		
	}
	
	private static class LogFreqSimilarityMeasure implements SameDomainSimilarityMeasure {

		private Counter<String> freqs = new Counter<String>();
		
		public LogFreqSimilarityMeasure(Counter<String> freqs) {
			this.freqs = freqs;
		}
		
		public double getSimilarity(String word1, String word2) {
			double dist = Math.abs(Math.log(freqs.getCount(word1)) - Math.log(freqs.getCount(word2)));
			return -dist;
		}
		
	}

	private static class ContextSimilarityMeasure implements SameDomainSimilarityMeasure {
		
//		private CounterMap<String,String> contexts = new CounterMap<String,String>(new MapFactory.IdentityHashMapFactory(), new MapFactory.IdentityHashMapFactory());
		private CounterMap<String,String> contexts = new CounterMap<String,String>();
		
		public ContextSimilarityMeasure(List<List<String>> docs, Indexer<String> wordsToExtract, List<String> featWords, int windowSize, ContextType contextType) {
			
			for (List<String> doc: docs) {
				for (int i=0; i < doc.size(); ++i) {
					String word = doc.get(i);
					//counts.incrementCount(word, 1.0);

					// only look at words in domain/codomain
					if (!wordsToExtract.contains(word)) {
						continue;
					}

					for (int z=-windowSize; z <= windowSize; ++z) {
						if (z == 0 || i+z < 0 || i+z >= doc.size()) { continue; }
						String neighbor = doc.get(i+z);
						// only count features for words in feature words set
						if (!featWords.contains(neighbor)) {
						continue;
						}
						String f = null;
//						for (ContextType contextType: ContextType.values()) {
						switch (contextType) {
						case POSITIONAL:
							if (Math.abs(z) < 2) {
								f = String.format("%d-%s-%s", z, neighbor,contextType.toString());
								break;
							}
						case DIRECTIONAL:
							f = String.format("%s-%s-%s", z > 0 ? "left" : "right", neighbor,contextType.toString());
							break;
						case UNDISTINGUISHED:							
							f = String.format("%s-%s",neighbor,contextType.toString());
							break;
						default:
							throw new Error("Bad context type.");
						}
						contexts.incrementCount(word, Interners.stringInterner.intern(f), 1.0);
//						}
					}
				}
			}
			contexts.normalize();
		}
		
		public double getSimilarity(String word1, String word2) {
			double L1 = 0.0;
			Counter<String> context1 = contexts.getCounter(word1);
			Counter<String> context2 = contexts.getCounter(word2);
			Set<String> feats = new HashSet<String>();
			feats.addAll(context1.keySet());
			feats.addAll(context2.keySet());
			for (String feat : feats) {
				L1 += Math.abs(context1.getCount(feat) - context2.getCount(feat));
			}
			return -L1;
		}
	}


	public SimilarityKernelFeatureExtractor(List<List<String>> docs, Indexer<String> wordsToExtract, List<String> featWords, Counter<String> wordCounts, Options opts) {
		
//		SameDomainSimilarityMeasure simMeas = new ContextSimilarityMeasure(docs, wordsToExtract, featWords, opts.windowSize, opts.contextType);
		SameDomainSimilarityMeasure simMeas = new LogFreqSimilarityMeasure(wordCounts);
//		SameDomainSimilarityMeasure simMeas = new LogUniqueContextCountSimilarityMeasure(docs, wordsToExtract, featWords, opts.uniqueWindowSize, opts.uniqueContextType);
		
		featManager = new FeatureManager();		
		sparseFeatureCounts = new CounterMap<String, Feature>(new MapFactory.HashMapFactory(), new MapFactory.IdentityHashMapFactory());
		for (String word : wordsToExtract) {
			for (String featWord : featWords) {
				Feature feat = featManager.getFeature("SIM_FEAT_"+featWord);
				sparseFeatureCounts.incrementCount(word, feat, simMeas.getSimilarity(word, featWord));
			}
		}
		
	}


	public Counter<Feature> getFeatures(String x) {
		return sparseFeatureCounts.getCounter(x);
	}

}
