package inducer;

public class GreedyMatchingExtractor implements MatchingExtractor {

	public int[] extractMatching(double[][] matchingPotentials) {
		int n =  matchingPotentials.length;
		int[] matching = new int[n];
		for (int i=0; i < n; ++i) {
			int argMax = -1;
			double max = Double.NEGATIVE_INFINITY;
			for (int j=0; j < n; ++j) {
				if (matchingPotentials[i][j] > max) {
					argMax = j;
					max = matchingPotentials[i][j];
				}
			}
			assert argMax > -1;
			matching[i] = argMax;			
		}
		return matching;
	}

}
