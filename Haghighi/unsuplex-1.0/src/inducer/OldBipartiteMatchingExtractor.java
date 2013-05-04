package inducer;

import java.util.HashSet;
import java.util.Set;

import canco.BipartiteMatcher;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.PriorityQueue;
import fig.basic.Indexer;

public class OldBipartiteMatchingExtractor<S,T> {
	
	boolean bipartite = true;
	int guessK = 0;
	int topK = Integer.MAX_VALUE;
	
	public OldBipartiteMatchingExtractor() {
		
	}
	
	public void setBipartite(boolean bipartite) {
		this.bipartite = bipartite;
	}
	

	public void setGuessK(int guessK) {
		this.guessK = guessK;
	}
	
	public void setBipartiteTopK(int topK) {
		this.topK = topK;
	}
	
	private void pruneMatchingMatrix(double[][] matchScores) {
		if (topK == Integer.MAX_VALUE) { return; }
		for (int i=0; i < matchScores.length; ++i) {
			Counter<Integer> scores = new Counter<Integer>();
			for (int j=0; j < matchScores[i].length; ++j) {
				scores.setCount(j, matchScores[i][j]);
			}
			PriorityQueue<Integer> pq = scores.asPriorityQueue();
			Set<Integer> valid = new HashSet<Integer>();
			for (int k=0; k < topK && pq.hasNext(); ++k) {
				valid.add(pq.next());
			}
			for (int j=0; j < matchScores[i].length; ++j) {
				if (!valid.contains(j)) {
					matchScores[i][j] = Double.NEGATIVE_INFINITY;
				}
			}
		}
	}

	public CounterMap<S,T> extractMatching(Indexer<S> lefts, Indexer<T> rights, double[][] matchScores) {		
		CounterMap<S, T> finalMatchings = new CounterMap<S, T>();
		if (bipartite) {
			assert lefts.size() == rights.size();
			pruneMatchingMatrix(matchScores);
			BipartiteMatcher matcher = new BipartiteMatcher(lefts.size());
			for (int i=0; i < lefts.size(); ++i) {
				for (int j=0; j < rights.size(); ++j) {
					matcher.setWeight(i, j, matchScores[i][j]);
				}
			}
						
			int[] matchingIndices = matcher.getMatching();
			for (int i=0; i < matchingIndices.length; ++i) {
				int j = matchingIndices[i];
				S left = lefts.getObject(i);
				T right = rights.getObject(j);
				finalMatchings.setCount(left, right, matchScores[i][j]);
			}			
		}
		else {
			CounterMap<S,T> rawMatchings = new CounterMap<S, T>();
			for (int i=0; i < lefts.size(); ++i) {
				S left = lefts.getObject(i);
				for (int j=0; j < rights.size(); ++j) {
					T right = rights.getObject(j);
					rawMatchings.setCount(left, right, matchScores[i][j]);
				}								
			}
			for (S left: rawMatchings.keySet()) {
				Counter<T> counts = rawMatchings.getCounter(left);
				PriorityQueue<T> pq = counts.asPriorityQueue();
				for (int k=0; k < guessK && pq.hasNext(); ++k) {
					double score = pq.getPriority();
					T right = pq.next();
					finalMatchings.setCount(left, right, score);
				}
			}
			
		}		
		return finalMatchings;
	}

}
