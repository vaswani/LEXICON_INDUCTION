package inducer;

import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.Pair;

public class EditDistanceInducer implements DictionaryInducer {
	
	private static final double cutoffDist = Double.POSITIVE_INFINITY;
	private double maxDist = Double.NEGATIVE_INFINITY;

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		int n = domWords.size();
		double[][] distMat = new double[n][n];
		for (int i=0; i < domWords.size(); ++i) {
			for (int j=0; j < codomWords.size(); ++j) {
				double dist = editDistance(domWords.getObject(i), codomWords.getObject(j));
//				if (dist > maxDist) maxDist = (double)dist;
				distMat[i][j] = -(double)dist; 
			}
		}
		
//		double[][] matching = new double[n][n];
		
		// normalize
		// TODO: is this crazy?
//		CombinedInducer.addScalarMult(matching, distMat, (-1.0)/maxDist);
//		CombinedInducer.pointwiseAddScalar(matching, 1.0);
		return distMat;
	}
	
	
//	List<String> domWords ;
//	List<String> codomWords ;
	
	public static double editDistance(String s, String t) {
		int d[][]; // matrix
		int n; // length of s
		int m; // length of t
		int i; // iterates through s
		int j; // iterates through t
		char s_i; // ith character of s
		char t_j; // jth character of t
		int cost; // cost
		// Step 1
		n = s.length();
		m = t.length();
		if (n == 0) {
			return m;
		}
		if (m == 0) {
			return n;
		}
		d = new int[n + 1][m + 1];
		// Step 2
		for (i = 0; i <= n; i++) {
			d[i][0] = i;
		}
		for (j = 0; j <= m; j++) {
			d[0][j] = j;
		}
		// Step 3
		for (i = 1; i <= n; i++) {
			s_i = s.charAt(i - 1);
			// Step 4
			for (j = 1; j <= m; j++) {
				t_j = t.charAt(j - 1);
				// Step 5
				if (s_i == t_j) {
					cost = 0;
				} else {
					cost = 1;
				}
				// Step 6
				d[i][j] = SloppyMath
						.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
			}
		}

		// Step 7
		// return d[n][m];
		double editDist = d[n][m];
//		double normEditDist = editDist / ((double) Math.max(s.length(), t.length()));
		return editDist;
	}

	public void setSeedMapping(CounterMap<String, String> ignored) {
		// intentionally blank
	}


	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		throw new UnsupportedOperationException("Representations undefined on this inducer.");
	}


	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		
	}

}
