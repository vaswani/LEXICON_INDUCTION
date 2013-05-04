package inducer;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.PriorityQueue;

import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;

public class HardEMInducer implements DictionaryInducer {

	DictionaryInducer inducer ;
	CounterMap<String, String> initSeedMapping ;
	Indexer<String> domWords, codomWords ;
	final int iters;
	final double seedSizeFactor;
	MatchingExtractor matchingExtractor;
	Evaluator callback;
	
	public HardEMInducer(DictionaryInducer inducer, Indexer<String> domWords, Indexer<String> codomWords, int iters, double seedSizeFactor, MatchingExtractor matchingExtractor,  Evaluator callback) {
		this.domWords = domWords;
		this.codomWords = codomWords;
		this.inducer = inducer;
		this.iters = iters;
		this.seedSizeFactor = seedSizeFactor;
		this.matchingExtractor = matchingExtractor;
		this.callback = callback;
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords,Indexer<String> codomWords) {
		return inducer.getMatchingMatrix(domWords, codomWords);
	}

	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		return inducer.getRepresentations(domWords, codomWords);
	}

	public void setSeedMapping(CounterMap<String, String> initSeedMapping) {
		this.initSeedMapping = initSeedMapping;	
		CounterMap<String, String> seedMapping = new CounterMap<String, String>();
		seedMapping.incrementAll(initSeedMapping);				
		for (int i=0; i < iters; ++i) {
			inducer.setSeedMapping(seedMapping);
			double[][] matchingScores = inducer.getMatchingMatrix(domWords, codomWords);
			CounterMap<String, String> guess = MatchingProcessor.getMatching(matchingExtractor, matchingScores, domWords, codomWords);
			CounterMap<String, String> newSeedMapping = new CounterMap<String, String>();
			newSeedMapping.incrementAll(initSeedMapping);
			newSeedMapping.incrementAll(guess);
			seedMapping = newSeedMapping;
//			LogInfo.logs("Iter %d result\n----------\n%s\n", i+1, callback.eval(guess));
		}
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		
	}

}
