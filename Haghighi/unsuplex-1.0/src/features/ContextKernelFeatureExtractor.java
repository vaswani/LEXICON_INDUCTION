package features;

import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.util.*;
import fig.basic.LogInfo;
import inducer.LinearSparseKernel.KernelFeatureExtractor;

public class ContextKernelFeatureExtractor implements KernelFeatureExtractor<String> {

	public static enum ContextType {POSITIONAL, DIRECTIONAL, UNDISTINGUISHED}

	private FeatureManager featManager;
	private CounterMap<String,Feature> sparseFeatureCounts;

	public ContextKernelFeatureExtractor(Iterator<List<String>> docs, List<String> wordsToExtract, List<String> featWords, int windowSize, ContextType contextType, int numSentences) {

		featManager = new FeatureManager();		
		sparseFeatureCounts = new CounterMap<String, Feature>(new MapFactory.HashMapFactory(), new MapFactory.IdentityHashMapFactory());
		int count = 0;
		LogInfo.logs("Extracting Context Features");
		while (docs.hasNext()) {
			double coef = 1.0; 
			
			if (++count > numSentences) {
				int divsor = numSentences / 2;
				int d = (count - numSentences) / divsor;
				coef = Math.pow(0.5, d+1);
			}
			
			List<String> doc = docs.next();
			for (int i=0; i < doc.size(); ++i) {
				String word = doc.get(i);

				// only look at words in domain/codomain			
				for (int z=-windowSize; z <= windowSize; ++z) {					
					if (z == 0 || i+z < 0 || i+z >= doc.size()) { continue; }
					String neighbor = doc.get(i+z);;

					// only count features for words in feature words set
					if (!featWords.contains(neighbor)) {
						continue;
					}
					String f = null;
					double weight = 0.0;
					switch (contextType) {
					case POSITIONAL:
						if (Math.abs(z) < 2) {
							f = String.format("%d-%s-%s", z, neighbor,contextType.toString());
							weight = 0.0;
							break;
						}
						break;
					case DIRECTIONAL:
						f = String.format("%s-%s-%s", z > 0 ? "left" : "right", neighbor,contextType.toString());
						weight = 0.5;
						break;
					case UNDISTINGUISHED:							
						f = String.format("%s-%s",neighbor,contextType.toString());
						weight = 1.0;
						break;
					default:
						throw new Error("Bad context type.");
					}
					Feature feat = featManager.getFeature(f);
					sparseFeatureCounts.incrementCount(word, feat, coef * weight);

				}
			}
		}

		for (String word: sparseFeatureCounts.keySet()) {
			Counter<Feature> vec = sparseFeatureCounts.getCounter(word);
			double sum = 0.0;
			for (Feature feat: vec.keySet()) {
				double x = vec.getCount(feat);
				sum += x*x;
			}
			double len = (sum > 0.0 ? Math.sqrt(sum) : 0.0);
			for (Feature feat: vec.keySet()) {
				double x = vec.getCount(feat);
				vec.setCount(feat, x/len);
			}			
		}

	}

	public Counter<Feature> getFeatures(String x) {
		return sparseFeatureCounts.getCounter(x);
	}
	
	
	

}
