package inducer;

import java.util.Map;

import edu.berkeley.nlp.util.Counter;
import features.Feature;
import kernelcca.Kernel;


public class LinearSparseKernel<T> implements Kernel<T> {
	
	public static interface KernelFeatureExtractor<T>  {
		Counter<Feature> getFeatures(T x);
	}
	
	KernelFeatureExtractor<T> featureExtractor;
	
	public LinearSparseKernel(KernelFeatureExtractor<T> featureExtractor) {
		this.featureExtractor = featureExtractor;
	}
	
	
	private double vecLen(Counter<Feature> featVec) {
		double sumSquared = 0.0;
		for (Map.Entry<Feature, Double> entry: featVec.getEntrySet()) {
			sumSquared += entry.getValue() * entry.getValue();
		}
		double len = Math.sqrt(sumSquared);
		return len; 
	}
	
	public double dot(T x, T y) {
		Counter<Feature> xFeatures = featureExtractor.getFeatures(x);
		Counter<Feature> yFeatures = featureExtractor.getFeatures(y);
		
		double xLen = 1.0;
		double yLen = 1.0;
		if (xLen == 0 || yLen == 0) {
			return 0.0;
		}
		
		double result = 0.0;
		if (xFeatures.size() < yFeatures.size()) {
			for (Map.Entry<Feature, Double> entry: xFeatures.getEntrySet()) {
				result += entry.getValue() * yFeatures.getCount(entry.getKey()) / (xLen * yLen);
			}
		} else {
			for (Map.Entry<Feature, Double> entry: yFeatures.getEntrySet()) {
				result += entry.getValue() * xFeatures.getCount(entry.getKey()) / (xLen * yLen);
			}
		}
		
		return result;
	}
	
	
	
}
