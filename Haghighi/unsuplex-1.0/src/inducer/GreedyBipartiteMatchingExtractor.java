package inducer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;

public class GreedyBipartiteMatchingExtractor implements MatchingExtractor {

	public int[] extractMatching(double[][] matchingPotentials) {
		int n = matchingPotentials.length;
		List<Prediction> preds = new ArrayList<Prediction>(n*n);
		for (int i=0; i < n; ++i) { 
			for (int j=0; j < n; ++j) {
				Prediction pred = new Prediction(i,j, matchingPotentials[i][j]);
				preds.add(pred);
			}
		}
		Collections.sort(preds);
		BitSet leftSet = new BitSet(n);
		BitSet rightSet = new BitSet(n);		
		int[] matching = new int[n];
		
		for (Prediction pred: preds) {
			if (leftSet.get(pred.i) || rightSet.get(pred.j)) {
				continue;
			}
			matching[pred.i] = pred.j;
			leftSet.set(pred.i);
			rightSet.set(pred.j);
		}		
		return matching;
	}
	
	private static class Prediction implements Comparable<Prediction> {
		int i, j ;
		double score ;
		public Prediction(int i, int j, double score) {
			super();
			this.i = i;
			this.j = j;
			this.score = score;
		}		
		public int compareTo(Prediction other) {			  
			 if(other.score > this.score) return 1;
			 if(other.score == this.score) return 0;
			 return -1;
		}
		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + i;
			result = prime * result + j;
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
			if (getClass() != obj.getClass())
				return false;
			final Prediction other = (Prediction) obj;
			if (i != other.i)
				return false;
			if (j != other.j)
				return false;
			if (Double.doubleToLongBits(score) != Double
					.doubleToLongBits(other.score))
				return false;
			return true;
		}
		public String toString() {
			return String.format("(%d,%d,%.5f)",i,j,score);
		}
	}

}
