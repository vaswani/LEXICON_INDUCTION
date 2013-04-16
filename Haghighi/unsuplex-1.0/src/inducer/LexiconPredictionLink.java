/**
 * 
 */
package inducer;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.PriorityQueue;

class LexiconPredictionLink {
	String domWord, codomWord ;
	double score ;
	public LexiconPredictionLink(String domWord, String codomWord, double score) {
		super();
		this.domWord = domWord;
		this.codomWord = codomWord;
		this.score = score;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
		+ ((codomWord == null) ? 0 : codomWord.hashCode());
		result = prime * result
		+ ((domWord == null) ? 0 : domWord.hashCode());
		long temp;
		temp = Double.doubleToLongBits(score);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		final LexiconPredictionLink other = (LexiconPredictionLink) obj;
		if (codomWord == null) {
			if (other.codomWord != null)
				return false;
		} else if (!codomWord.equals(other.codomWord))
			return false;
		if (domWord == null) {
			if (other.domWord != null)
				return false;
		} else if (!domWord.equals(other.domWord))
			return false;
		if (Double.doubleToLongBits(score) != Double
				.doubleToLongBits(other.score))
			return false;
		return true;
	}
	
	static CounterMap<String, String> getTopKPredictions(CounterMap<String, String> matching,int topK) {
		Counter<LexiconPredictionLink> predScores = new Counter<LexiconPredictionLink>();
		for (String dom: matching.keySet()) {
			Counter<String> codomScores = matching.getCounter(dom);
			for (String codom: codomScores.keySet()) {
				double score = codomScores.getCount(codom);
				LexiconPredictionLink pred = new LexiconPredictionLink(dom,codom, score);	
				predScores.setCount(pred, score);
			}
		}
		PriorityQueue<LexiconPredictionLink> pq = predScores.asPriorityQueue();
		CounterMap<String, String> topKCounts = new CounterMap<String, String>();		
		for (int k=0; k < topK && pq.hasNext(); ++k) {
			LexiconPredictionLink pred = pq.next();
			topKCounts.setCount(pred.domWord, pred.codomWord, 1.0);//predScores.getCount(pred));
		}
		return topKCounts;
	}
}