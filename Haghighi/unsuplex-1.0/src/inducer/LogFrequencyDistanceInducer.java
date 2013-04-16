package inducer;

import java.util.Map;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.PriorityQueue;
import fig.basic.Indexer;
import fig.basic.Pair;

public class LogFrequencyDistanceInducer implements DictionaryInducer {
	
//	private static final double cutoffDist = 0.2;
//	private double maxDist = Double.NEGATIVE_INFINITY;
	private Counter<String> domCounts;
	private Counter<String> codomCounts;
	
	private <T> void logCounts(Counter<T> counts) {
		for (Map.Entry<T, Double> entry : counts.getEntrySet()) {
			Double val = entry.getValue();
			entry.setValue(Math.log(val));
		}
	}
	
	
	public LogFrequencyDistanceInducer(Counter<String> domainWordCounts, Counter<String> codomainWordCounts) {
		this.domCounts = domainWordCounts;
		logCounts(domCounts);		
		this.codomCounts = codomainWordCounts;
		logCounts(codomCounts);
	}
	
//	private Counter<String> getLogRanksCounter(Counter<String> wordCounts) {
//		Counter<String> ranksCounter = new Counter<String>();
//		PriorityQueue<String> wordQueue = wordCounts.asPriorityQueue();
//		double pos = 1.0;
//		while (wordQueue.hasNext()) {
//			ranksCounter.setCount(wordQueue.next(), Math.log(pos));
//			pos++;
//		}
//		return ranksCounter;
//	}

	double getDomLogFreq(String word) {
		return domCounts.getCount(word);
	}
	
	double getCodomLogFreq(String word) {
		return codomCounts.getCount(word);
	}

	private double getLogRankDistance(String domWord, String codomWord) {
		double domLogFreq = getDomLogFreq(domWord);
		double codomLogFreq = getCodomLogFreq(codomWord);
		double dist = Math.abs(domLogFreq-codomLogFreq);
//		if (dist > maxDist) maxDist = dist;
	 	return dist;
	}
	
	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		int n = domWords.size();
		double[][] distMat = new double[n][n];
		for (int i=0; i < domWords.size(); ++i) {
			for (int j=0; j < codomWords.size(); ++j) {
				double dist = (double)getLogRankDistance(domWords.getObject(i), codomWords.getObject(j));
				distMat[i][j] = dist; 
			}
		}
		
		double[][] matching = new double[n][n];
		
		// normalize
		// TODO: is this crazy?
		CombinedInducer.addScalarMult(matching, distMat, (-1.0));
		CombinedInducer.pointwiseAddScalar(matching, 1.0);
		return matching;
	}

	public void setSeedMapping(CounterMap<String, String> ignored) {
		// intentionally blank
	}


	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		throw new UnsupportedOperationException("Representations undefined on this inducer.");
	}


	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		
	}


}
