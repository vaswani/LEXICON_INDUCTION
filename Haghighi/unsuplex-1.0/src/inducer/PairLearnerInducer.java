package inducer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import kernelcca.CachingVectorRepresenter;
import kernelcca.ConstantGaussianUnitModel;
import kernelcca.ConstantUnitModel;
import kernelcca.Kernel;
import kernelcca.MultivariateGaussianUnitModel;
import kernelcca.PairModel;
import kernelcca.ProbCCAModel;
import kernelcca.UnitModel;
import kernelcca.VectorRepresenter;
import kernelcca.IncompleteCholeskyDecomposition;

import edu.berkeley.nlp.mapper.MapWorker;
import edu.berkeley.nlp.mapper.MapWorkerFactory;
import edu.berkeley.nlp.mapper.Mapper;
import fig.basic.*;
import static fig.basic.LogInfo.*;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

public class PairLearnerInducer implements DictionaryInducer {
	
	PairModel pairModel;
	VectorRepresenter<String> domRepn;
	VectorRepresenter<String> codomRepn;
	Kernel<String> domKernel;
	Kernel<String> codomKernel;
	UnitModel domUnitModel ;
	UnitModel codomUnitModel ;
	
	public PairLearnerInducer(Kernel<String> domKernel, Kernel<String> codomKernel) {
		this.domKernel = domKernel;
		this.codomKernel = codomKernel;
	}
	
	private UnitModel getUnitModel() {
		switch (DictionaryInducerTester.opts.unitModelType) {
		case NONE: return new ConstantUnitModel();
		case GAUSSIAN: return new ConstantGaussianUnitModel();
		case MULTI_GAUSSIAN: return new MultivariateGaussianUnitModel();
		}
		throw new RuntimeException();
	}
	
	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {

		LogInfo.track("Filling matching matrix: (%d, %d) words", domWords.size(), codomWords.size());
		double[][] matchingMatrix = new double[domWords.size()][codomWords.size()];
		

//		LogInfo.track("Getting linear representations");
//			Pair<double[][],double[][]> pair = getRepresentations(domWords, codomWords);
//			double[][] domRepns = pair.getFirst();		
//			double[][] codomRepns = pair.getSecond();
//		LogInfo.end_track();
		
		LogInfo.track("Forming simScores");
			for (int i=0; i < domWords.size(); ++i) {
				double[] domVec = domRepn.getRepn(domWords.getObject(i));//domRepns[i];			
				for (int j=0; j < codomWords.size(); ++j) {
					double[] codomVec = codomRepn.getRepn(codomWords.getObject(j));//codomRepns[j];
					double sim = pairModel.getScore(domVec, codomVec);
					double domNull = domUnitModel.getScore(domVec);
					double codomNull = codomUnitModel.getScore(codomVec);
					matchingMatrix[i][j] = sim - domNull - codomNull;
				}
			}
		LogInfo.end_track();

		LogInfo.end_track();
		return matchingMatrix;
	}
	
	private VectorRepresenter<String> getRepn(String[] words, Kernel<String> kernel) {			
		//double tau = DictionaryInducerTester.pccaOptions.tau;
		VectorRepresenter<String> repn = new IncompleteCholeskyDecomposition<String>(kernel,words, 0.001);
		return new CachingVectorRepresenter<String>(repn);
	}
	
	private String[] convertToArray(List<String> lst) {
		String[] arr = new String[lst.size()];
		for (int i = 0; i < lst.size(); i++) {
			arr[i] = lst.get(i);
		}
		return arr;
	}

	public void setSeedMapping(CounterMap<String, String> seedMapping) {		
		List<String> domWords = new ArrayList<String>();
		List<String> codomWords = new ArrayList<String>();
		Iterator<Pair<String,String>> it = seedMapping.getPairIterator();
		while (it.hasNext()) {
			Pair<String, String> p = it.next();
			domWords.add(p.getFirst());
			codomWords.add(p.getSecond());
		}
		domRepn = getRepn(convertToArray(domWords), domKernel);
		codomRepn = getRepn(convertToArray(codomWords), codomKernel);
		
		
		pairModel = new ProbCCAModel(DictionaryInducerTester.pccaOptions);
		
		LogInfo.track("setSeedMapping");
		List<Pair<double[],double[]>> pairs = new ArrayList<Pair<double[],double[]>>();
		List<double[]> domVecs = new ArrayList<double[]>();
		List<double[]> codomVecs = new ArrayList<double[]>();
		
		for (String dom: seedMapping.keySet()) {
			Counter<String> c = seedMapping.getCounter(dom);
			double[] domVec = domRepn.getRepn(dom);
			for (String codom: c.keySet()) {				
				double[] codomVec = codomRepn.getRepn(codom);
				Pair<double[],double[]> p = Pair.newPair(domVec,codomVec);
				domVecs.add(domVec);
				codomVecs.add(codomVec);
				pairs.add(p);
			}
		}
		pairModel.learn(pairs);		
		domUnitModel = getUnitModel();
		domUnitModel.learn(domVecs);
		codomUnitModel = getUnitModel();
		codomUnitModel.learn(codomVecs);			

		LogInfo.end_track();
	}


	public Pair<double[][], double[][]> getRepresentations(final Indexer<String> domWords, final Indexer<String> codomWords) { 
		final double[][]	 domRepns = new double[domWords.size()][];
		final double[][] 	 codomRepns = new double[codomWords.size()][];

		class Worker extends MapWorker<Integer> {
			@Override
			public void map(Integer item) {
				int i = item;
				domRepns[i] = domRepn.getRepn(domWords.getObject(i));
				codomRepns[i] = codomRepn.getRepn(codomWords.getObject(i));
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
		// Right Thing to Do
//		VectorRepresenter<String> domRepn = new IncompleteCholeskyDecomposition<String>(domKernel,convertToArray(domWords),DictionaryInducerTester.pccaOptions.tau);
//		VectorRepresenter<String> codomRepn = new IncompleteCholeskyDecomposition<String>(codomKernel,convertToArray(codomWords),DictionaryInducerTester.pccaOptions.tau);
	}


}
