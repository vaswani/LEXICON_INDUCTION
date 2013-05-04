package inducer;

import inducer.Evaluator.Prediction;

import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;

public class SlidingWindowInducer implements DictionaryInducer {
	DictionaryInducer base ;
	Indexer<String> domWords ;
	Indexer<String> codomWords ;
	int numWindows ;
	MatchingExtractor matchingExtractor;
	CounterMap<String, String> cachedGuess = new CounterMap<String, String>();
	Evaluator eval;
	
	public SlidingWindowInducer(int numWindows, DictionaryInducer base, MatchingExtractor matchingExtractor, Evaluator eval) {
		this.numWindows = numWindows;
		this.base = base;
		this.matchingExtractor = matchingExtractor;
		this.eval = eval;
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		throw new RuntimeException();
	}

	public Pair<double[][], double[][]> getRepresentations(
			Indexer<String> domWords, Indexer<String> codomWords) {
		throw new UnsupportedOperationException();
	}

	public void setSeedMapping(CounterMap<String, String> initSeedMapping) {
		CounterMap<String, String> seedMapping = new CounterMap<String, String>();
		seedMapping.incrementAll(initSeedMapping);
		for (int w=0; w < numWindows; ++w) {
			int start = (int) (((double) w / (double) numWindows) * domWords.size());
			int end = (int) (((double) (w+1) / (double) numWindows) * codomWords.size());
			
			if (w > 0 && base instanceof IterativeInducer) {
				IterativeInducer iterInducer = (IterativeInducer) base;
				iterInducer.seedSizeFactor *= 0.50;
				iterInducer.iters = Math.max(0, iterInducer.iters-2);
				LogInfo.logs("Modified IterativeInducer to %.3f and %d",iterInducer.seedSizeFactor, iterInducer.iters);
			}
			
			LogInfo.track("SlidingWindowInducer.setSeedMapping: current wingow [%d,%d]",start,end);
				Indexer<String> windowDomWords = new Indexer<String>(domWords.subList(start,end));
				Indexer<String> windowCodomWords = new Indexer<String>(codomWords.subList(start,end));
				base.setWords(windowDomWords, windowCodomWords);
				base.setSeedMapping(seedMapping);			
				double[][] matchingMatrix = base.getMatchingMatrix(windowDomWords,windowCodomWords);
				CounterMap<String, String> guess = MatchingProcessor.getMatching(matchingExtractor, matchingMatrix, windowDomWords,windowCodomWords);
				cachedGuess.incrementAll(guess);
				int numEntriesToAdd = (int) (Math.pow(0.5,w+1) * guess.totalSize());
				for (Prediction pred: Evaluator.getSortedPrediction(guess).subList(0,numEntriesToAdd))  {
					seedMapping.incrementCount(pred.srcWord, pred.trgWord, 1.0);	
				}
				LogInfo.logs("Adding %d new entries",numEntriesToAdd);
			LogInfo.end_track();
			
			LogInfo.track("SlidingWindowInducer.eval");
				LogInfo.logs(eval.eval(guess, windowDomWords, windowCodomWords));
			LogInfo.end_track();
		}
	}
	
	
	
	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		this.domWords = domWords;
		this.codomWords = codomWords;
	}
	
}
