package inducer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import inducer.DictionaryInducerTester.Options;
import wordAlignment.*;
import wordAlignment.distortion.DistortionModel;
import wordAlignment.distortion.IBMModel1;
import wordAlignment.distortion.StateMapper;
import wordAlignment.distortion.StringDistanceModel;
import wordAlignment.distortion.StateMapper.EndsStateMapper;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.SentencePairReader;
import edu.berkeley.nlp.mt.SentencePairReader.PairDepot;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;
import fig.basic.String2DoubleMap;

public class BerkeleyAlignerInducer implements DictionaryInducer {
	static final int NUM_ITERS = 5;
	private EMWordAligner aligner;
	private Iterable<SentencePair> sentences ;
	private int maxNumParallelSentences ;

	private static Collection<SentencePair> getParallelSentences(Pair<List<List<String>>,List<List<String>>> corpora, int num) {
		List<List<String>> dom =  corpora.getFirst().subList(0, num);
		List<List<String>> codom =  corpora.getSecond().subList(0, num);
		Collection<SentencePair> sentPairs = new ArrayList<SentencePair>();
		for (int i=0; i < dom.size(); ++i) {
			SentencePair sp = new SentencePair(-1,null,dom.get(i), codom.get(i));
			sentPairs.add(sp);
		}
		return sentPairs;
	}

	public BerkeleyAlignerInducer(Options opts, NewBitext bitext) {
		// Get training data
		Pair<List<List<String>>,List<List<String>>> corpora = bitext.getBaseCorpus();
		this.maxNumParallelSentences = opts.maxNumParallelSentences;
		int num = Math.min(opts.numParallelSentences, maxNumParallelSentences);
		sentences = getParallelSentences(corpora, num);		
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		double[][] match = new double[domWords.size()][codomWords.size()];
		for (int i=0; i < domWords.size(); ++i) {
			for (int j=0; j < codomWords.size(); ++j) {
				double prob = aligner.getParams().transProbs.get(domWords.getObject(i), codomWords.getObject(j), 1.0e-20);				
				match[i][j] = Math.log(prob);
			}
		}
		return match;
	}

	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException();
	}

	public void setSeedMapping(CounterMap<String, String> seedMapping) {
		// Lexical Prior
		String2DoubleMap lexicalPrior = new String2DoubleMap();		
		Iterator<Pair<String, String>> it = seedMapping.getPairIterator();
		while (it.hasNext()) {
			Pair<String, String> p = it.next();
			lexicalPrior.incr(p.getFirst(), p.getSecond(), 1.0);
		}

		// Train 5 Iters of Model 1
		DistortionModel distModel = new IBMModel1(); 
		distModel.initUniform();		
		aligner = new EMWordAligner(distModel.getSpsFactory(), null, false);		
		aligner.initializeModel("", distModel, false, false, sentences);		
		aligner.setLexicalPrior(lexicalPrior);
		// Train model
		aligner.train(sentences, NUM_ITERS);
		StrCondProbTable lexProbs = aligner.getParams().transProbs;

		// HMM
		StateMapper mapper = new EndsStateMapper();
		distModel = new StringDistanceModel(mapper);
		distModel.initUniform();
		aligner = new EMWordAligner(distModel.getSpsFactory(), null, false);
		aligner.trainingCache = distModel.getTrainingCache();
		aligner.setModel(new Model<DistortionModel>("Norm", false, distModel));
		aligner.getParams().transProbs = lexProbs;
		// Train model
		aligner.train(sentences, NUM_ITERS);
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// intentionally blank
	}

}
