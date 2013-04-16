package inducer;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.Pair;

public class CombinedInducer implements DictionaryInducer {
	
	private boolean doGridSearch = false;
	private List<DictionaryInducer> inducers = new ArrayList<DictionaryInducer>();
	private List<Double> weights = new ArrayList<Double>();
	private List<Double> gridLower = new ArrayList<Double>();
	private List<Double> gridUpper = new ArrayList<Double>();
	private List<Double> gridIntervals = new ArrayList<Double>();
	
	
	public CombinedInducer(DictionaryInducer...inducers) {
		for (DictionaryInducer i : inducers) {
			addInducer(i, 1.0);
		}
	}
	
	public void addInducer(DictionaryInducer i, double w) {
		inducers.add(i);
		weights.add(w);
	}
	
	public void setWeights(List<Double> weights) {
		assert weights.size() == inducers.size();
		this.weights = weights;
	}
	
	
	// TODO: Should be on simplex?
	public void setGrid(List<Double> gridLower, List<Double> gridUpper, List<Double> gridIntervals) {
		this.gridLower = gridLower;
		this.gridUpper = gridUpper;
		this.gridIntervals = gridIntervals;
	}
	
	private void gaussianNormalize(double[][] m) {
//		System.out.println("g normalizing");
		double sum = 0.0;
		double sumSquared = 0.0;
		int n = 0;
		for (double[] row: m) { for (double x: row) { sum += x; sumSquared += x*x; n++; }}
		double mu = sum / n;
		double sigma = Math.max(1.0e-4,Math.sqrt( sumSquared/n - mu*mu ));
		for (double[] row: m) {
			pointwiseAddScalar(row, -mu);
			DoubleArrays.scale(row, 1.0/sigma);
		}
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {						
		// get matrix dimensions from first inducer
		int m = domWords.size();
		int n = codomWords.size();		
		double[][] result = new double[m][n];
		for (int index=0; index<inducers.size(); index++) {
			double[][] matchingMatrix = inducers.get(index).getMatchingMatrix(domWords, codomWords);
			double[][]	newMatchingMatrix = new double[m][n];
			//gaussianNormalize(newMatchingMatrix);
			addScalarMult(newMatchingMatrix, matchingMatrix, 1.0);			
			addScalarMult(result, newMatchingMatrix, weights.get(index));
		}						
		return result;
	}

	public void setSeedMapping(CounterMap<String, String> seedMapping) {
		for (DictionaryInducer i : inducers) {
			i.setSeedMapping(seedMapping);
		}
		if (doGridSearch) {
			weights = new ArrayList<Double>(gridLower);			
		}
	}

	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		throw new UnsupportedOperationException("Representations undefined on this inducer.");
	}

	public static void pointwiseAddScalar(double[] vect, double c) {
		for (int index=0; index < vect.length; index++) {
			vect[index] += c;
		}
	}
	
	public static void pointwiseAddScalar(double[][] mat, double c) {
		for (int index=0; index < mat.length; index++) {
			pointwiseAddScalar(mat[index], c);
		}
	}
	
	public static void addScalarMult(double[] sum, double[] vect, double c) {
		for (int index=0; index < sum.length; index++) {
			sum[index] += vect[index]*c;
		}
	}
	
	public static void addScalarMult(double[][] sum, double[][] mat, double c) {
		for (int index=0; index < sum.length; index++) {
			addScalarMult(sum[index], mat[index], c);
		}
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		for (DictionaryInducer inducer: inducers) {
			inducer.setWords(domWords, codomWords);
		}
	}
	
}
