package inducer;

import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;

public class MatchingProcessor  {
	public static <S,T> CounterMap<S,T> getMatching(MatchingExtractor matchingExtractor, double[][] matchingPotentials, Indexer<S> left, Indexer<T> right) {
		int[] matching = matchingExtractor.extractMatching(matchingPotentials);
		CounterMap<S,T> cm = new CounterMap<S, T>();
		for (int i=0; i < left.size(); ++i) {
			int j = matching[i];
			cm.setCount(left.getObject(i), right.getObject(j), matchingPotentials[i][j]);
		}
		return cm;
	}
}
