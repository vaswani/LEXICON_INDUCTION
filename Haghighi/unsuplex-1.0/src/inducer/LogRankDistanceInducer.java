package inducer;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.PriorityQueue;
import fig.basic.Indexer;
import fig.basic.Pair;

public class LogRankDistanceInducer implements DictionaryInducer {
	
	private static final double cutoffDist = 0.2;
	private double maxDist = Double.NEGATIVE_INFINITY;
	private Counter<String> domRanks;
	private Counter<String> codomRanks;
	
	
	public LogRankDistanceInducer(Counter<String> domainWordCounts, Counter<String> codomainWordCounts) {
		this.domRanks = getLogRanksCounter(domainWordCounts);
		this.codomRanks = getLogRanksCounter(codomainWordCounts);
	}
	
	private Counter<String> getLogRanksCounter(Counter<String> wordCounts) {
		Counter<String> ranksCounter = new Counter<String>();
		PriorityQueue<String> wordQueue = wordCounts.asPriorityQueue();
		double pos = 1.0;
		while (wordQueue.hasNext()) {
			ranksCounter.setCount(wordQueue.next(), Math.log(pos));
			pos++;
		}
		return ranksCounter;
	}

	double getDomLogRank(String word) {
		return domRanks.getCount(word);
	}
	
	double getCodomLogRank(String word) {
		return codomRanks.getCount(word);
	}

	private double getLogRankDistance(String domWord, String codomWord) {
		double domRank = getDomLogRank(domWord);
		double codomRank = getCodomLogRank(codomWord);
		double dist = Math.min(Math.abs(domRank-codomRank), cutoffDist);
		if (dist > maxDist) maxDist = dist;
		return dist;
	}
	
	private double getLogRankDistanceNew(String domWord, String codomWord) {
		double domRank = getDomLogRank(domWord);
		double codomRank = getCodomLogRank(codomWord);
		double dist = (domRank-codomRank) * (domRank-codomRank);		
//		dist = Math.min(dist, cutoffDist);
		return dist;
	}
	
	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		int n = domWords.size();
		double[][] distMat = new double[n][n];
		for (int i=0; i < domWords.size(); ++i) {
			for (int j=0; j < codomWords.size(); ++j) {
				double dist = (double)getLogRankDistanceNew(domWords.getObject(i), codomWords.getObject(j));
				distMat[i][j] = dist; 
			}
		}
		
		double[][] matching = new double[n][n];
		
		// normalize
		// TODO: is this crazy?
		CombinedInducer.addScalarMult(matching, distMat, (-1.0)/maxDist);
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
