package inducer;


import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;

public class RepresentationMatchingExtractor {
	
	private double dist(double[] x, double[] y) {
		double sum = 0.0;
		for (int i=0; i < x.length; ++i) {
			sum += Math.abs(x[i]-y[i]);
		}
		return sum;
	}
	
	public CounterMap<String, String> getMatching(double[][] domRepns, double[][] codomRepns, Indexer<String> domWords, Indexer<String> codomWords) {
		CounterMap<String, String> matching = new CounterMap<String, String>();
		
		// Layer Dom Words ontop of Codom Words
		List<double[]> datums = new ArrayList<double[]>();
		for (int i=0; i <  domWords.size(); ++i) {
			datums.add(domRepns[i]);
		}
		for (int i=0; i <  codomWords.size(); ++i) {
			datums.add(codomRepns[i]);
		}
		int K = 1;
		KMeans kmeans = new KMeans(1);
		int[] clusterIds = kmeans.cluster(datums);
		List<List<Integer>> clusterMembers = new ArrayList<List<Integer>>(K);
		for (int k=0; k < K; ++k) { clusterMembers.add(new ArrayList<Integer>()); }
		for (int i=0; i < clusterIds.length; ++i) {
			clusterMembers.get(clusterIds[i]).add(i);
		}
		for (int i=0; i < domWords.size(); ++i) {
			double minDist = Double.POSITIVE_INFINITY;
			int argMin = -1;
			for (int j:  clusterMembers.get(clusterIds[i])) {
				j -= domWords.size();
				if (j < 0) { continue; }
				double d = dist(domRepns[i], codomRepns[j]);
				if (d < minDist) { minDist = d; argMin = j; }
			}
			matching.setCount(domWords.getObject(i), codomWords.getObject(argMin), minDist);
		}
		return matching;
	}
}
