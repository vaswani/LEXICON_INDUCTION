package inducer;

import edu.berkeley.nlp.util.CounterMap;

import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;
import fig.exec.Execution;

public class IterativeInducer implements DictionaryInducer {

	DictionaryInducer inducer ;
	Indexer<String> domWords, codomWords ;
	int iters;
	double seedSizeFactor;
	MatchingExtractor matchingExtractor;
	Evaluator callback;
	boolean useWeightedBootstrap;

	public IterativeInducer(DictionaryInducer inducer, DictionaryInducerTester.Options opts) {
		this.inducer = inducer;
		this.iters = opts.iters;
		this.seedSizeFactor = opts.iterativeSeedFactor;
		this.matchingExtractor = opts.matchingExtractor;
		this.callback = opts.callback;
		this.useWeightedBootstrap = opts.useWeightedBootstrap;
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords,Indexer<String> codomWords) {
		return inducer.getMatchingMatrix(domWords, codomWords);
	}

	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		return inducer.getRepresentations(domWords, codomWords);
	}

	public void setSeedMapping(CounterMap<String, String> initSeedMapping) {

		
		LogInfo.track("IterativeInducer.setSeed");
		double topKPerIter =  (double) ((seedSizeFactor * domWords.size()) / (iters-1.0));
		LogInfo.logs("IterativeInducer: topKPerIter=%.3f",topKPerIter);
		CounterMap<String, String> topKLastIter = new CounterMap<String, String>();
		for (int i=0; i < iters; ++i) {							
			LogInfo.track("iter %d/%d",i+1,iters);
				int topK = (int) topKPerIter * (i+1);
				CounterMap<String, String> seedMapping = new CounterMap<String, String>();
				// all initial seed
				seedMapping.incrementAll(initSeedMapping);
				// top k most confident things from last round
				seedMapping.incrementAll(topKLastIter);
				inducer.setSeedMapping(seedMapping);
	
				if (i+1 < iters) {
					LogInfo.track("IterativeInducer: fillingMatchingScores");
					double[][] matchingScores = inducer.getMatchingMatrix(domWords, codomWords); 
					LogInfo.end_track();

					LogInfo.track("extractingMatching: " + matchingExtractor.getClass().getSimpleName());
					CounterMap<String, String> matching = MatchingProcessor.getMatching(matchingExtractor, matchingScores, domWords, codomWords);
					LogInfo.end_track();

					LogInfo.track("results for current iteration");
					LogInfo.logsForce(callback.eval(matching,domWords,codomWords));					
					LogInfo.end_track();

					topKLastIter = LexiconPredictionLink.getTopKPredictions(matching, topK);
					LogInfo.logs("TopKLastIter: %d", topKLastIter.size());
					assert topKLastIter.size() == topK;
				}
			LogInfo.end_track();
			LogInfo.end_track();
		}

		LogInfo.end_track();
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		this.domWords = domWords;
		this.codomWords = codomWords;
	}

}
