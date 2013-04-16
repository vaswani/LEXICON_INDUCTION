package inducer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import canco.BipartiteMatcher;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.PriorityQueue;
import fig.basic.LogInfo;

public class BipartiteMatchingExtractor implements MatchingExtractor {

	
	
	private void filterTopN(double[][] potentials, int N) {
		if (N == Integer.MAX_VALUE) { return ; }
		List<Set<Integer>> indicesToKeep = new ArrayList<Set<Integer>>();
		for (int i=0; i < potentials.length; ++i) {
			Counter<Integer> counts = new Counter<Integer>();
			for (int j=0; j < potentials[i].length; ++j) {
				counts.setCount(j, potentials[i][j]);
			}
			PriorityQueue<Integer> pq = counts.asPriorityQueue();
			Set<Integer> set = new HashSet<Integer>();			
			for (int k=0; k < N && pq.hasNext(); ++k) {
				set.add(pq.next());
			}
			indicesToKeep.add(set);
		}
		for (int i=0; i < potentials.length; ++i) {
			Set<Integer> set = indicesToKeep.get(i);
			for (int j=0; j < potentials[i].length; ++j) {
				if (!set.contains(j)) {
					potentials[i][j] = Double.NEGATIVE_INFINITY;
				}
			}
		}
	}
	
	public int[] extractMatching(double[][] matchingPotentials) {
		int n = matchingPotentials.length;
		LogInfo.track("biparteMatchExtract");
		
		// Do initial greedy match
//		int[] approxMatch = new GreedyBipartiteMatchingExtractor().extractMatching(matchingPotentials);		
		BipartiteMatcher matcher = new BipartiteMatcher(n);
//		filterTopN(matchingPotentials, 25);
//		matcher.setInitMatching(approxMatch);
		for (int i=0; i < n; ++i) { 
			for (int j=0; j < n; ++j) {
				if (Double.isNaN(matchingPotentials[i][j])) {
					LogInfo.logs("BipartiteMatchingExtractor.extractMatching: NaN Hack!");
					matchingPotentials[i][j] = Double.NEGATIVE_INFINITY;
				}
				matcher.setWeight(i, j, matchingPotentials[i][j]);
			}
		}
		int[] matching = matcher.getMatching();
		LogInfo.end_track();
		return matching;
	}

}
