package inducer;

import fig.basic.Indexer;
import fig.basic.Pair;

import edu.berkeley.nlp.util.CounterMap;

public interface DictionaryInducer {
	
	/**
	 * assert domWords.size() == codomWords.size()
	 */
	// slow one
	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) ;
		
	public void setSeedMapping(CounterMap<String, String> seedMapping) ;
	
	public void setWords(Indexer<String> domWords, Indexer<String> codomWords);

	public Pair<double[][],double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords);
}
