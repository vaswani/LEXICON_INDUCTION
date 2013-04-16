package inducer;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Pair;

public interface WeightedSeedInducer {
	public CounterMap<String, String> getSeed(Collection<Pair<List<String>, List<String>>> parallelSentences) ;
	
	public static class DiceSeedInducer implements WeightedSeedInducer {

		public CounterMap<String, String> getSeed(Collection<Pair<List<String>, List<String>>> parallelSentences) {
			Counter<String> leftCounts = new Counter<String>();
			Counter<String> rightCounts = new Counter<String>();
			CounterMap<String, String> jointCounts = new CounterMap<String, String>();
			for (Pair<List<String>, List<String>> pair : parallelSentences) {
				List<String> leftwords = pair.getFirst();
				List<String> rightWords = pair.getSecond();
				leftCounts.incrementAll(leftwords, 1.0);
				rightCounts.incrementAll(rightWords, 1.0);
				for (String lword : leftwords) {
					for (String rword : rightWords) {
						jointCounts.incrementCount(lword, rword, 1.0);
					}
				}
			}
			CounterMap<String, String> dice = new CounterMap<String, String>();
			Iterator<Pair<String, String>> it = jointCounts.getPairIterator();
			while (it.hasNext()) {
				Pair<String, String> p = it.next();
				double d = (2 * jointCounts.getCount(p.getFirst(),p.getSecond())) / (leftCounts.getCount(p.getFirst()) * rightCounts.getCount(p.getSecond()));
				dice.setCount(p.getFirst(), p.getSecond(), d);
			}
			dice.normalize();
			return dice;
		}
		
	}
	
	
}
