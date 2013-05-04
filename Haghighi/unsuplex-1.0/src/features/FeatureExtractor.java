package features;

import java.util.Collection;
import java.util.List;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

public abstract class FeatureExtractor {

	FeatureManager featManager = new FeatureManager();
	CounterMap<String, Feature>	sparseFeatureCounts ;
	double[] mus ;
	double[] sigmas ;

	public FeatureExtractor(List<List<String>> docs) {
		this.sparseFeatureCounts = new CounterMap<String, Feature>();
	}
	
	public abstract void extractFeatures(Collection<String> wordsToExtract);
	
	public FeatureManager getFeatureManager() {
		return featManager;
	}
	
	public CounterMap<String,Feature> getSparseFeatureCounter() {
		return sparseFeatureCounts;
	}
	
	public void gaussNormalizeFeatures() {
		mus = new double[featManager.getNumFeatures()];
		sigmas = new double[featManager.getNumFeatures()];

		for (int f=0; f < featManager.getNumFeatures(); ++f) {
			double sum = 0.0;
			double sumSquared = 0.0;
			Feature feat = featManager.getFeature(f);
			int numWords = sparseFeatureCounts.keySet().size();
			for (String word: sparseFeatureCounts.keySet()) {
				Counter<Feature> featCounts = sparseFeatureCounts.getCounter(word);
				double count = featCounts.getCount(feat); 
				sum += count;
				sumSquared += count * count;
			}
			double mean = sum / numWords;
			mus[f] = mean;
			double expectSquared = sumSquared / numWords;
			double sigma = expectSquared - mean*mean;
			sigmas[f] = sigma;				
		}
	}

	public double[] getProjectedVector(String word) {
		double[] vec =  new double[featManager.getNumFeatures()];
		Counter<Feature> featCounts = sparseFeatureCounts.getCounter(word);
		for (Feature feat: featCounts.keySet()) {
			vec[feat.getIndex()] = featCounts.getCount(feat);
		}
		if (mus != null && sigmas != null) {
			for (int f=0; f < vec.length; ++f) {
				vec[f] = (vec[f]-mus[f]) / sigmas[f];
			}
		}
		return vec;
	}

	public double[][] getFeaturesOnSeed(List<String> seeds) {
		double[][] mat =  new double[seeds.size()][];
		for (int i=0; i < seeds.size(); ++i) {
			String seed = seeds.get(i);
			mat[i] = getProjectedVector(seed);
		}
		return mat;
	}
}