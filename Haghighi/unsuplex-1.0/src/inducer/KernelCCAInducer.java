package inducer;

import java.util.ArrayList;
import java.util.List;

import kernelcca.Kernel;
import kernelcca.KernelCCA;
import kernelcca.VectorRepresenter;

import edu.berkeley.nlp.mapper.MapWorker;
import edu.berkeley.nlp.mapper.MapWorkerFactory;
import edu.berkeley.nlp.mapper.Mapper;
import fig.basic.*;
import static fig.basic.LogInfo.*;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

public class KernelCCAInducer implements DictionaryInducer {

	private KernelCCA.Options kccaOptions;	

	// Input
	private Kernel<String> domKernel;
	private Kernel<String> codomKernel;
	private Kernel<String> sharedKernel;
	private Kernel<String> solKernel ;
	public static CCAInducerOptions opts = new CCAInducerOptions();

	// Output
	private KernelCCA<String> kernelCCA;

	public static class CCAInducerOptions {
		@Option
		public boolean finalNorm = false;

		@Option
		public boolean useRankPrior = false;

		@Option
		public boolean useL2Dist = false;

		@Option
		public double penaltyCoef = 1.0;
		
		@Option
		public double penaltyAlpha = 1.05;
		
		@Option
		public boolean doPenaltyLog = true;
		
		@Option
		public double penaltyCounts = 10.0;
	}

	public KernelCCAInducer(KernelCCA.Options kccaOptions, Kernel<String> domKernel, Kernel<String> codomKernel, Kernel<String> sharedKernel) {
		this.kccaOptions = kccaOptions;
		this.domKernel = domKernel;
		this.codomKernel = codomKernel;
		this.sharedKernel = sharedKernel;

	}
	
	private double l2Dist(double[] x, double[] y) {
		double xNorm = opts.finalNorm ? vecNorm(x) : 1.0;
		double yNorm = opts.finalNorm ? vecNorm(y) : 1.0;
		if (xNorm == 0.0) { 
			xNorm = 1.0;
		}
		if (yNorm == 0.0) {
			yNorm = 1.0;
		}
		double sum = 0.0;
		for (int i=0; i < x.length; ++i) {
			sum += (x[i]/xNorm - y[i]/yNorm) * (x[i]/xNorm - y[i]/yNorm); 
		}
		return Math.sqrt(sum);
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {

		LogInfo.track("Filling matching matrix: (%d, %d) words", domWords.size(), codomWords.size());
		double[][] matchingMatrix = new double[domWords.size()][codomWords.size()];

		LogInfo.track("Getting linear representations");
		Pair<double[][],double[][]> pair = getRepresentations(domWords, codomWords);
		double[][] domRepns = pair.getFirst();
		double[][] codomRepns = pair.getSecond();
		LogInfo.logs("Each string is mapped to %d dimensions", domRepns[0].length);
		if(kccaOptions.verbose >= 2) {
			for(int i = 0; i < domWords.size(); i++)
				dbg("X: " + domWords.getObject(i) + " => " + Fmt.D(domRepns[i]));
			for(int i = 0; i < codomWords.size(); i++)
				dbg("Y: " + codomWords.getObject(i) + " => " + Fmt.D(codomRepns[i]));
		}
		LogInfo.end_track();

		LogInfo.track("Forming dot products");
		for (int i=0; i < domWords.size(); ++i) {
			String domWord = domWords.getObject(i);
			for (int j=0; j < codomWords.size(); ++j) {
				String codomWord = codomWords.getObject(j);
				double sim = 0.0; 
				if (kccaOptions.probabilistic) {
					sim = solKernel.dot(domWord, codomWord);
				}
				else if (opts.useL2Dist) {
					sim = -l2Dist(domRepns[i], codomRepns[j]);
				}
				else {
					for(int b = 0; b < domRepns[i].length; b++) {
						double weight = kernelCCA.getBasisWeight(b) ;					
						double contrib = weight * domRepns[i][b] * codomRepns[j][b];
						if (Double.isNaN(contrib) || Double.isInfinite(contrib)) {
							continue;
						}
						sim += contrib;
					}
					if (Double.isNaN(sim) || Double.isInfinite(sim)) {
						LogInfo.logs("NaN hack\n");
						sim = 0.0; 
					}
					if (opts.finalNorm) {
						double domNorm = vecNorm(domRepns[i]);
						double codomNorm = vecNorm(codomRepns[j]);
						if (domNorm > 0.0 && codomNorm > 0.0) {
							sim /= (domNorm*codomNorm);
						}
					}
				} 
				
				matchingMatrix[i][j] = sim;
				if(kccaOptions.verbose >= 2) {
					dbg(domWord + " " + codomWord + " => " + Fmt.D(sim));
				}
			}
		}

		// Matching Prior
		if (opts.useRankPrior) {
			for (int i=0; i < domWords.size(); ++i) {
				for (int j=0; j < codomWords.size(); ++j) {
					double curScore = matchingMatrix[i][j];
					double penalty = distortion(i, j);							
					matchingMatrix[i][j] = curScore + opts.penaltyCoef * penalty;
				}
			}
		}

		LogInfo.end_track();

		LogInfo.end_track();
		return matchingMatrix;
	}

//	private int distBucket(int d) { 
//		if (d == 0) { return 0; }
//		if (d < 5) { return 1; }
//		if (d < 10) { return 2; }
//		return 3;
//	}
//
//	private double harmonic(int n) {
//		double sum = 0.0;
//		for (int i=0; i < n; ++i) { sum += 1.0 / (1.0+i); }
//		return sum;
//	}

	private double distortion(int i, int j) {
		double alpha = opts.penaltyAlpha;
		int max = Math.max(i, j);
		int min = Math.min(i, j);
		double arg = (min+opts.penaltyCounts) / (max+opts.penaltyCounts);
		assert arg <= 1.0;
		double exponent = Math.log(arg);
		assert exponent <= 0.0;
		double penalty = Math.pow(alpha,exponent);
		if (opts.doPenaltyLog) { penalty = Math.log(penalty); }
		return penalty;
	}

	//	private void vecNormalize(double[] vec) {
	//		double sumsq = 0.0;
	//		for (double x: vec) { sumsq += x * x; }
	//		double len = Math.sqrt(sumsq);
	//		if (len == 0.0) { return ; }
	//		DoubleArrays.scale(vec, 1.0/len);
	//	}

	private double vecNorm(double[] vec) {
		double norm = 0.0;
		for (double x: vec) {
			norm += x*x;
		}
		return Math.sqrt(norm);
	}


	public void setSeedMapping(CounterMap<String, String> seedMapping) {
		kernelCCA = new KernelCCA<String>(kccaOptions);
		LogInfo.track("KernelCCA.setSeedMapping");
		List<Pair<String, String>> pairs = new ArrayList<Pair<String, String>>();
		List<Double> weightList = new ArrayList<Double>();
		for (String dom: seedMapping.keySet()) {
			Counter<String> c = seedMapping.getCounter(dom);
			for (String codom: c.keySet()) {
				Pair<String, String> p = Pair.newPair(dom, codom);
				pairs.add(p);
				weightList.add(seedMapping.getCount(dom, codom));
			}
		}
		int i =0;		
		double[] weights = new double[weightList.size()];
		for (double x: weightList) { weights[i++] = x; }
		kernelCCA.setData(pairs, domKernel, codomKernel, sharedKernel);
		kernelCCA.setWeights(weights);
		solKernel = kernelCCA.solve();
		LogInfo.end_track();
	}


	public Pair<double[][], double[][]> getRepresentations(final Indexer<String> domWords, final Indexer<String> codomWords) {
		final double[][]	 domRepns = new double[domWords.size()][];
		final double[][] 	 codomRepns = new double[codomWords.size()][];
		final VectorRepresenter<String> domRepn = kernelCCA.getXRepresentation();
		final VectorRepresenter<String> codomRepn = kernelCCA.getYRepresentation();

		class Worker extends MapWorker<Integer> {
			@Override
			public void map(Integer item) {
				int i = item;
				domRepns[i] = domRepn.getRepn(domWords.getObject(i));
				//				vecNormalize(domRepns[i]);
				codomRepns[i] = codomRepn.getRepn(codomWords.getObject(i));
				//				vecNormalize(codomRepns[i]);
			}			
		}

		MapWorkerFactory<Integer> fact = new MapWorkerFactory<Integer>() {
			public MapWorker<Integer> newMapWorker() {
				// TODO Auto-generated method stub
				return new Worker();
			} };

			int n = domWords.size();	
			assert domWords.size() == codomWords.size() ;
			List<Integer> items = new ArrayList<Integer>();
			for (int i=0; i < n; ++i) { items.add(i); }
			Mapper<Integer> mapper = new Mapper<Integer>(fact);
			mapper.doMapping(items);
			return Pair.newPair(domRepns, codomRepns);
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub

	}


}
