package inducer;

import Jama.Matrix;
import io.BitextCorpusExtractor;
import io.POSTagPrefixes;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import kernelcca.*;

import translex.BilingualLexicon;
import translex.TranslationLexicon;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Lists;
import edu.berkeley.nlp.util.PriorityQueue;
import edu.berkeley.nlp.util.Stats;
import edu.berkeley.nlp.util.Triple;
import features.ContextKernelFeatureExtractor.ContextType;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Pair;
import fig.exec.Execution;

public class DictionaryInducerTester implements Runnable {

	public static class Options {
		@Option
		public boolean disjointCorpora = false;

		@Option
		public String wordsPOS = "noun";

		@Option
		public String contextPOS = "noun";

		@Option
		public String uniqueContextPOS = "all";

		@Option
		public int numContextWords = Integer.MAX_VALUE;

		@Option(gloss="")
		public int numWords = 50;

		@Option()
		public int windowSize = 4;

		@Option()
		public int uniqueWindowSize = 1;

		@Option()
		public ContextType contextType = ContextType.UNDISTINGUISHED;

		@Option()
		public ContextType uniqueContextType = ContextType.POSITIONAL;

		public static enum EStepType {
			EXACT_BIPARTITE, APPROX_BIPARTITE, GREEDY
		};
		@Option
		public EStepType eStepType = EStepType.EXACT_BIPARTITE;

		@Option
		public int seedSize = 10;

		@Option
		public int threshLevels = 6;

		@Option
		public double orthoWeight = 1.0;

		@Option
		public double contextWeight = 1.0;

		@Option
		public double similarityWeight = 0.0;

		@Option
		public boolean verbose = false;

		@Option(required=true)
		public String basePath ;

		@Option()
		public boolean combinedInducer = false;

		@Option()
		public boolean justEditDist = false;

		@Option(gloss="Use SCA")
		public boolean useSharedStringKernel = false;

		@Option()
		public double stringKernelAlpha = 0.9;

		@Option()
		public int iters = 10;

		@Option()
		public double kernelWeight = 1.0;

		@Option()
		public double editWeight = 0.0;

		@Option()
		public double rankWeight = 0.0;

		@Option
		public double uniqueWeight = 0.0;

		@Option
		public boolean justLogRank = false;

		@Option
		public boolean useLogCountHack = false;

		@Option
		public boolean useLogUniqueContextHack = false;

		@Option
		public double iterativeSeedFactor = 0.5;

		@Option
		public int numParallelSentences = 0;

		@Option
		public boolean useHardEM = false;

		@Option
		public boolean useWeightedBootstrap = false;

		@Option
		public boolean useAlignmentInducer = false;


		@Option
		public int maxNumParallelSentences = 5000;

		@Option
		public int maxAdditionalSents = 0;

		@Option(gloss="for fast debugging.")
		public int maxSentences = Integer.MAX_VALUE;

		@Option
		public boolean useDiceSeedMapping = false;

		@Option
		public boolean printCanonicalPDF = false;

		@Option
		public int numCanonicalPDFWords = 6;

		@Option
		public boolean useEqualWordHack = false;

		@Option
		public boolean editDistSeed = false;

		@Option
		public boolean useLogLinear = false;

		@Option
		public int numWindows = 0;

		@Option
		public boolean doRankTest = false;

		@Option
		public boolean writeSeedDict = false;

		@Option
		public boolean useProbCCA = true;

		public static enum UnitModelType {
			NONE, GAUSSIAN, MULTI_GAUSSIAN
		}
		
		@Option
		public UnitModelType unitModelType = UnitModelType.NONE;

        @Option
        public boolean useGraphs = false;

        @Option
        public double lambda = 0;

		public Evaluator callback ; 
		public BitextCorpusExtractor bitextCorpusExtractor ;
		public MatchingExtractor matchingExtractor;

	}

	public static DictionaryInducer getInducer(Pair<Indexer<String>,Indexer<String>> words, CounterMap<String,String> seedMapping, NewBitext bitext, Options opts) {
		if (opts.justEditDist) {
			return  new EditDistanceInducer();
		}

		DictionaryInducer inducer = null;

		if (opts.justLogRank) {
			inducer = new LogFrequencyDistanceInducer(bitext.getWordCounts().getFirst(), bitext.getWordCounts().getSecond());
			return inducer;
		}

    // Fancy new thing
    if(opts.useLogLinear)
      return new LogLinearInducer(llOptions, words.getFirst(), words.getSecond(), seedMapping, bitext);

		// get KernelCCAInducer

		if (opts.useAlignmentInducer) {
			inducer = new BerkeleyAlignerInducer(opts, bitext);
			return inducer;
		}
		Triple<Kernel<String>, Kernel<String>, Kernel<String>> bestKernels = BestKernel.getBestKernel(words, bitext, opts);
		Kernel<String> domKernel = bestKernels.getFirst();
		Kernel<String> codomKernel = bestKernels.getSecond();
		Kernel<String> sharedKernel = bestKernels.getThird();


		DictionaryInducer ccaInducer = null;
		if (opts.useProbCCA) {
			//			PairModel pairModel = new ProbCCAModel(pccaOptions);
            if (opts.useGraphs) {
                int N = opts.numWords;
                Matrix domG = new Matrix(N, N);
                Matrix codomG = new Matrix(N, N);
                ccaInducer = new GraphPairLearnerInducer(domKernel, codomKernel, domG, codomG, opts.lambda);
            } else {
			    ccaInducer = new PairLearnerInducer(domKernel,codomKernel);
            }

		} else {
			ccaInducer  = new KernelCCAInducer(kccaOptions, domKernel, codomKernel, sharedKernel);
		}

		if (opts.combinedInducer) {
			LogFrequencyDistanceInducer logFreqInducer = new LogFrequencyDistanceInducer(bitext.getWordCounts().getFirst(), bitext.getWordCounts().getSecond());
			EditDistanceInducer editDistInducer = new EditDistanceInducer();
			// unique context counts inducer
			Set<POSTagPrefixes.POSTag> contextTags = POSTagPrefixes.getTagSet(opts.uniqueContextPOS);			
			Pair<List<List<String>>,List<List<String>>> reducedCorpora = bitext.getReducedBaseCorpus(contextTags);
			LogUniqueContextCountInducer uniqueContextInducer = new LogUniqueContextCountInducer(reducedCorpora, words, opts.uniqueWindowSize, opts.contextType);

			// get CombinedInducer (kernel, edit, rank, unique)
			List<Double> weights = Lists.newList(opts.kernelWeight, opts.editWeight, opts.rankWeight, opts.uniqueWeight);
			CombinedInducer combInducer = new CombinedInducer(ccaInducer, editDistInducer, logFreqInducer, uniqueContextInducer);
			combInducer.setWeights(weights);
			inducer = combInducer;	
		} else {
			inducer = ccaInducer;
		}

		if (opts.iters > 0) {
			// get the iterative version of usedInducer
			if (opts.useHardEM) {
				throw new RuntimeException();
				//				inducer = new HardEMInducer(inducer, words.getFirst(), words.getSecond(), opts.iters,opts.iterativeSeedFactor, opts.matchingExtractor, opts.callback);
			} else {
				inducer = new IterativeInducer(inducer, opts);
			}
		}

		if (opts.useLogCountHack) {
			inducer = new LogCountHackInducer(inducer, bitext.getWordCounts().getFirst(), bitext.getWordCounts().getSecond());
		}

		if (opts.useEqualWordHack) {
			inducer = new EqualWordHackInducer(inducer);	
		}

		//		if (opts.useLogUniqueContextHack) {
		//			Set<POSTagPrefixes.POSTag> contextTags = POSTagPrefixes.getTagSet(opts.uniqueContextPOS);
		//			opts.bitextProcessor.setPOSTags(contextTags);
		//			Pair<List<List<String>>,List<List<String>>> reducedCorpora = opts.bitextProcessor.getReducedCorproa();
		//			inducer = new LogUniqueContextCountHackInducer(inducer, reducedCorpora, words, opts.uniqueWindowSize, opts.contextType);
		//		}

		if (opts.numWindows > 0) {
			inducer = new SlidingWindowInducer(opts.numWindows, inducer, opts.matchingExtractor, opts.callback);
		}

		return inducer;

	}

	static Options opts = new Options();
	static KernelCCA.Options kccaOptions;
	static ProbCCAModel.Options pccaOptions;
	static LogLinearInducer.Options llOptions = new LogLinearInducer.Options();


	public void processOptions(Options opts) {
		opts.bitextCorpusExtractor = new BitextCorpusExtractor.NoOpExtractor();
		if (opts.disjointCorpora) {
			LogInfo.logss("Disjoint corpora.");
			opts.bitextCorpusExtractor = new BitextCorpusExtractor.MixedExtractor(opts.numParallelSentences);
		}
		switch (opts.eStepType) {
		case EXACT_BIPARTITE: 
			opts.matchingExtractor = new BipartiteMatchingExtractor();
			break;
		case APPROX_BIPARTITE:
			opts.matchingExtractor = new GreedyBipartiteMatchingExtractor();
			break;			
		case GREEDY:
			opts.matchingExtractor = new GreedyMatchingExtractor();
			break;
		}
		LogInfo.logss("Matching Extractor: %s\n", opts.matchingExtractor.getClass().getSimpleName());
	}

	private static <T> Collection<T> fromIterator(Iterator<T> it) {
		Collection<T> coll= new ArrayList<T>();
		while (it.hasNext()) {
			T t = it.next();
			coll.add(t);			
		}
		return coll;
	}


	private void rankTest(Indexer<String> domWords, Counter<String> domCounter, Indexer<String> codomWords, Counter<String> codomCounter, TranslationLexicon transLex) {
		Stats stats = new Stats(true);
		Counter<String> penaltyScores = new Counter<String>();
		for (String domWord: domWords) {
			for (String codomWord: codomWords) {
				if (transLex.getImage(domWord).contains(codomWord)) {
					int i = domWords.indexOf(domWord);
					int j = codomWords.indexOf(codomWord);
					int rankDiff = Math.abs(i-j);
					int max = Math.max(i,j);
					int min = Math.min(i,j);
					double alpha = 1.1;
					double arg = (min+10.0) / (max+10.0);
					double exponent = Math.log(arg);
					assert exponent <= 0.0;
					double penalty = Math.log(Math.pow(alpha, -exponent));									
					String penaltyStr = String.format("%s (%d with count %.3f) - %s (%d with count %.3f) => penalty:%.3f",domWord,i, domCounter.getCount(domWord),codomWord,j, codomCounter.getCount(codomWord), penalty);
					//					LogInfo.logs(penaltyStr);
					penaltyScores.setCount(penaltyStr, -penalty);
					//					assert !Double.isNaN(rankDiff) && !Double.isInfinite(rankDiff);
					stats.observe(penalty);
				}
			}
		}		
		for (String p: penaltyScores.getSortedKeys()) {
			LogInfo.logs(p);
		}
		LogInfo.logs("rank stats: " + stats.toString());
		LogInfo.logs("quantiles: " + stats.getQuantiles(300));
		Execution.finish();
		System.exit(0);
	}

	public void run() {
		LogInfo.track("DictionaryInducerTest");

		// get bitext
		processOptions(opts);
		NewBitext bitext = new NewBitext(opts.basePath,opts.bitextCorpusExtractor,opts.maxSentences,opts.maxAdditionalSents);

		if (!opts.disjointCorpora) {
			opts.numParallelSentences = bitext.getBaseCorpus().getFirst().size();
		}

		// print run info
		LogInfo.logs("Base Path: %s",opts.basePath);
		LogInfo.logs("POS: " + opts.wordsPOS);
		LogInfo.logs("Context POS: " + opts.contextPOS);

		// get domain and codomain
		final Pair<Indexer<String>,Indexer<String>> words = bitext.getMostCommonWordsInOrder(opts.numWords+opts.seedSize, POSTagPrefixes.getTagSet(opts.wordsPOS));
		Execution.putOutput("domWords", words.getFirst());
		Execution.putOutput("codomWords", words.getSecond());
		LogInfo.logs("Num Words: %d %d", words.getFirst().size(), words.getSecond().size());
		assert words.getFirst().size() >= opts.numWords && words.getSecond().size() >= opts.numWords;

		// get gold lexicon		
		// TODO Allow multiple translations
		final BilingualLexicon transLex = bitext.getLexicon();

		// get seed mapping
		final CounterMap<String, String> seedMapping = getSeedMapping(words, transLex, opts);
		LogInfo.logss("Size of Seed: %d", seedMapping.size());
		Pair<Indexer<String>,Indexer<String>> reducedWords = removeSeedWords(words, seedMapping);
		words.setFirst(reducedWords.getFirst()); words.setSecond(reducedWords.getSecond());
		Execution.putOutput("seedPairs", fromIterator(seedMapping.getPairIterator()));

		// Rank Test
		//		if (opts.doRankTest) {
		//			rankTest(words.getFirst(), bitext.domainCounts, words.getSecond(), bitext.codomainCounts, transLex);
		//		}

		// set eval callback
		Evaluator evaluator = new Evaluator(transLex,opts.threshLevels,opts.verbose); 
		opts.callback = evaluator;

		// run inducer, find best matching
		DictionaryInducer inducer = getInducer(words, seedMapping, bitext, opts);
		LogInfo.logss("inducer: " + inducer.getClass().getSimpleName());

		LogInfo.track("inducer.setWords %d",words.getFirst().size());
		inducer.setWords(words.getFirst(), words.getSecond());
		LogInfo.end_track();

		LogInfo.track("inducer.setSeedMapping");
		inducer.setSeedMapping(seedMapping);
		LogInfo.end_track();

		//		LogInfo.track("printCanonicalPDF");
		// print canonical space representation
		//		if (opts.printCanonicalPDF ) {
		//			Pair<double[][],double[][]> reps = inducer.getRepresentations(words.getFirst(), words.getSecond());
		//			CanonicalSpacePDFPrinter canFirst2Printer = new CanonicalSpacePDFPrinter(new First2Representer2D());
		//			canFirst2Printer.writePDF(reps, words.getFirst(), words.getSecond(), transLex, seedMapping, opts.numCanonicalPDFWords, "first2CanonicalSpace.pdf");
		//			CanonicalSpacePDFPrinter canPCAPrinter = new CanonicalSpacePDFPrinter(new PCARepresenter2D(reps));
		//			canPCAPrinter.writePDF(reps, words.getFirst(), words.getSecond(), transLex, seedMapping, opts.numCanonicalPDFWords, "PCACanonicalSpace.pdf");
		//		}
		//		LogInfo.end_track();


		CounterMap<String, String> matching = null;

		// Super Hack!
		if (inducer instanceof SlidingWindowInducer) {
			matching = ((SlidingWindowInducer) inducer).cachedGuess;
		} else {
			LogInfo.track("inducer.getMatchingMatrix");
			double[][] matchingScores = inducer.getMatchingMatrix(words.getFirst(), words.getSecond());
			LogInfo.end_track();

			LogInfo.track("Computing matching");		
			matching = MatchingProcessor.getMatching(opts.matchingExtractor, matchingScores, words.getFirst(), words.getSecond());
		}

		LogInfo.logs("final matching: %s",evaluator.getSortedAnnotatedPrediction(matching));
		Execution.putOutput("final matching", evaluator.getSortedAnnotatedPrediction(matching));		
		writeOutputDictionary(matching, seedMapping);
		List<Pair<Double, Double>> precisionRecallPoints = evaluator.getPrecisionRecallPoints(matching, words.getFirst(), words.getSecond());
		writePrecisionRecallInformation(precisionRecallPoints);
		LogInfo.end_track();

		LogInfo.logs(opts.callback.eval(matching,words.getFirst(),words.getSecond()));

		LogInfo.end_track();
		Execution.finish();
	}

	private void writePrecisionRecallInformation(List<Pair<Double, Double>> pairs) {
		String dir = Execution.getVirtualExecDir();
		String path = dir + "/" + "pr-points.txt";
		LogInfo.logs("Writting P/R points to %s",path);
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path),Charset.forName("utf8")));
			for (Pair<Double, Double> p : pairs) {					
				writer.write(String.format("%.3f\t%.3f\n",p.getFirst(),p.getSecond()));
			}
			writer.flush(); writer.close();
		} catch (Exception e) { System.exit(0);} 
	}

	private void writeOutputDictionary(CounterMap<String, String> guess, CounterMap<String,String> seedMapping) {
		String dir = Execution.getVirtualExecDir();
		String path = dir + "/" + "dict.txt";
		LogInfo.logs("Writting dictionary to %s",path);
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path),Charset.forName("utf8")));
			for (String dom: guess.keySet()) {
				Counter<String> counts = guess.getCounter(dom);
				for (Map.Entry<String, Double> entry: counts.getEntrySet()) {
					writer.write(String.format("%s\t%s\t%.5f\n",dom,entry.getKey(),entry.getValue()));
				}
			}
			if (opts.writeSeedDict) {
				for (String dom: seedMapping.keySet()) {
					Counter<String> counts = seedMapping.getCounter(dom);
					for (Map.Entry<String, Double> entry: counts.getEntrySet()) {
						writer.write(String.format("%s\t%s\t%.5f\n",dom,entry.getKey(),100.0));
					}
				}
			}
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	//	private void statTest(Bitext bitext,
	//			final FunctionalTranslationLexicon transLex) {
	//		Stats stats = new Stats();
	//		for (Pair<String, String> p: transLex.getTranslationPairs()) {
	//			double domLogCount = Math.log(bitext.getWordCounts().getFirst().getCount(p.getFirst()));
	//			double codomLogCount = Math.log(bitext.getWordCounts().getSecond().getCount(p.getSecond()));
	//			double diff = Math.abs(domLogCount-codomLogCount);
	//			stats.observe(diff);
	//		}
	//		System.out.println(stats);
	//		System.exit(0);
	//	}

	public static void main(String[] args) {
		//		GlobalArgs.globalArgs = args;
		//		opts = new Options();
		kccaOptions = new KernelCCA.Options();
		pccaOptions = new ProbCCAModel.Options();
		Execution.run(args, new DictionaryInducerTester(), opts, 
				"kernelCCA", kccaOptions,
				"probCCA",pccaOptions,
				"ccaInducer",KernelCCAInducer.opts,
        "logLinear", llOptions,
				"evalOptions",Evaluator.opts,
				"bitextOptions",NewBitext.opts);
	}

	private static Pair<Indexer<String>, Indexer<String>> removeSeedWords(Pair<Indexer<String>, Indexer<String>> words, CounterMap<String, String> seedMapping) {
		Collection<String> reducedDomWords = new ArrayList<String>();
		Collection<String> reducedCodomWords = new ArrayList<String>();
		Set<String> seedDomWords = new HashSet<String>();
		Set<String> seedCodomWords = new HashSet<String>();
		Iterator<Pair<String,String>> it = seedMapping.getPairIterator();
		StringBuilder builder = new StringBuilder();
		while (it.hasNext()) {
			Pair<String,String> p = it.next();
			seedDomWords.add(p.getFirst());
			assert words.getFirst().contains(p.getFirst());
			seedCodomWords.add(p.getSecond());
			assert words.getSecond().contains(p.getSecond());
			builder.append(p); if (it.hasNext()) { builder.append(" , "); }			
		}
		LogInfo.logs("SeedPairs: %s",builder);
		assert seedDomWords.size() == seedCodomWords.size() : String.format("%d %d",seedDomWords.size(),seedCodomWords.size());

		reducedDomWords.addAll(words.getFirst());
		reducedDomWords.removeAll(seedDomWords);

		reducedCodomWords.addAll(words.getSecond());
		reducedCodomWords.removeAll(seedCodomWords);

		assert reducedDomWords.size() == reducedCodomWords.size();
		return Pair.newPair(new Indexer<String>(reducedDomWords), new Indexer<String>(reducedCodomWords));
	}

	private static CounterMap<String, String> getEditDistSeedMapping(Pair<Indexer<String>,Indexer<String>> words, Options opts) {
		CounterMap<String, String> seedMapping = new CounterMap<String, String>();
		EditDistanceInducer editDistInducer = new EditDistanceInducer();
		double[][] matchingScores = editDistInducer.getMatchingMatrix(words.getFirst(), words.getSecond());
		CounterMap<String, String> matching = MatchingProcessor.getMatching(new BipartiteMatchingExtractor(), matchingScores, words.getFirst(), words.getSecond());
		Counter<Pair<String,String>> predScores = new Counter<Pair<String,String>>();
		Iterator<Pair<String,String>> pairIt = matching.getPairIterator();
		while (pairIt.hasNext()) {
			Pair<String, String> p = pairIt.next();
			predScores.setCount(p, matching.getCount(p.getFirst(),p.getSecond()));
		}
		PriorityQueue<Pair<String,String>> pq = predScores.asPriorityQueue();
		for (int k=0; k < opts.seedSize && pq.hasNext(); ++k) {
			Pair<String,String> p = pq.next();
			seedMapping.setCount(p.getFirst(), p.getSecond(), 1.0);
		}
		return seedMapping;
	}

	private static CounterMap<String, String> getSeedMapping(Pair<Indexer<String>,Indexer<String>> words, BilingualLexicon transLex, Options opts) {
		if (opts.editDistSeed) {
			return getEditDistSeedMapping(words, opts);
		} 
		return getGoldSeedMapping(words, transLex, opts);
	}

	//	private static CounterMap<String, String> getGoldSeedMapping(Pair<Indexer<String>,Indexer<String>> words, BilingualLexicon transLex, Options opts) {
	//		CounterMap<String, String> seedMapping = new CounterMap<String, String>();
	//		Set<String> seenCodom = new HashSet<String>();
	//		for (String domWord: words.getFirst()) {
	//			String codomWord = transLex.getImage(domWord);			
	////			if (codomWord != null && words.getSecond().contains(codomWord)) {				
	//			if (codomWord != null && words.getSecond().contains(codomWord) && !seenCodom.contains(codomWord)) {
	//				seedMapping.setCount(domWord, codomWord, 1.0);				
	//				seenCodom.add(codomWord);
	//				if (seedMapping.size() >= opts.seedSize) { 
	//					break;
	//				}
	//			}
	//		}
	//		return seedMapping;
	//	}

	private static CounterMap<String, String> getGoldSeedMapping(Pair<Indexer<String>,Indexer<String>> words, BilingualLexicon transLex, Options opts) {
		CounterMap<String, String> seedMapping = new CounterMap<String, String>();
		Set<String> seenCodom = new HashSet<String>();
		for (String domWord: words.getFirst()) {
			Set<String> codomWords = transLex.getImage(domWord);			
			//			if (codomWord != null && words.getSecond().contains(codomWord)) {
			//			boolean foundTranslation = false;
			for (String codomWord: codomWords) {				
				if (codomWord != null && words.getSecond().contains(codomWord) && !seenCodom.contains(codomWord)) {
					seedMapping.setCount(domWord, codomWord, 1.0);				
					seenCodom.add(codomWord);
					if (seedMapping.size() >= opts.seedSize) { 
						return seedMapping;
					} else {
						break;
					}
				}
			}
		}
		return seedMapping;
	}



	//	public static String gradeDictionary(Collection<String> domain, Collection<String> codomain, Collection<String> seedDomain, Collection<String> seedCodomain, CounterMap<String,String> guess, TranslationLexicon goldLexicon, int gradeKMappings, double maxSim, double minSim, int threshLevels, boolean verbose) {
	//		StringBuffer result = new StringBuffer();
	//
	//		double bestFMeas = 0.0;
	//		String bestResult = null;
	//		double bestThresh = -1.0;
	//		for (int i=threshLevels-1; i >= 0; i--) {
	//			double percent = (double) i / (double) (threshLevels);
	//			double thresh = minSim + percent * (maxSim-minSim);
	//			result.append(String.format("%n%n--- Sim threshold: %.3f, percent: %.3f, max sim: %.3f, min sim: %.3f ---%n%n", thresh, percent, maxSim, minSim));
	//			Pair<Double, String> res = gradeDictionary(domain, codomain, seedDomain, seedCodomain, guess, goldLexicon, gradeKMappings, thresh, verbose);
	//			double fmeas = res.getFirst();
	//			String threshResult = res.getSecond();
	//			if (fmeas > bestFMeas) {
	//				bestFMeas = fmeas;
	//				bestResult = threshResult;
	//				bestThresh = thresh;
	//			} 
	//
	//			result.append(threshResult);
	//		}
	////		result.append(String.format("%n%n------ Best Sim threshold: %.3f\n", bestThresh));
	////		result.append(bestResult);
	//		Execution.putOutput("bestF1", bestFMeas);
	//
	//		return result.toString();
	//	}

	public static String gradeDictionaryNew(Indexer<String> domain, Indexer<String> codomain, CounterMap<String, String> seedMapping, final CounterMap<String,String> guess, TranslationLexicon goldLexicon, int threshLevels, boolean verbose) {
		StringBuilder result = new StringBuilder();
		List<Pair<String, String>> predictions = new ArrayList<Pair<String,String>>();
		for (String dom: domain) {
			if (seedMapping.containsKey(dom)) { continue; }
			Counter<String> domGuesses = guess.getCounter(dom);
			for (String codom: domGuesses.keySet()) {
				predictions.add(Pair.newPair(dom, codom));
			}
		}
		Collections.sort(predictions, new Comparator<Pair<String, String>>() {

			private double get(Pair<String, String> p) { return guess.getCount(p.getFirst(), p.getSecond()); }

			public int compare(Pair<String, String> s, Pair<String, String> t) {
				return (int) (get(t) - get(s));
			}

		});
		int possible = 0;
		for (String dom: domain) {
			if (seedMapping.containsKey(dom)) { continue; }
			if (!goldLexicon.getDomain().contains(dom)) {
				continue;
			}
			Set<String> img = goldLexicon.getImage(dom);
			if (img.isEmpty()) { continue; }
			boolean intersect = false;
			for (String codom: img) { 
				if (codomain.contains(codom)) { intersect = true; }
			}; 
			if (intersect) { possible++ ; }
		}

		String bestResult = null;
		double bestF1 = 0.0;

		for (int i=0; i < threshLevels; ++i) {
			double fraction = (i+1) / ((double) threshLevels);
			int toIndex = (int) (fraction * predictions.size());
			List<Pair<String, String>> curPreds = predictions.subList(0, toIndex);
			int correct=0, guessed=0;
			for (Pair<String, String> p: curPreds) {				
				guessed++;
				if (!goldLexicon.getDomain().contains(p.getFirst())) {
					continue;
				}
				Set<String> goldImg = goldLexicon.getImage(p.getFirst()); 
				double sim = guess.getCount(p.getFirst(),p.getSecond());
				if (goldImg.contains(p.getSecond())) {
					correct++;
					if (verbose) {
						result.append(String.format("*correct: %s => %s %.3f\n",p.getFirst(), p.getSecond(),sim));
					}
				} else {
					if (verbose) {
						result.append(String.format("*wrong: %s => %s %.3f\n",p.getFirst(), p.getSecond(),sim));
					}
				}
			}
			double prec = correct / (double) guessed;
			double recall = correct/ (double) possible;
			double f1 = 2.0*prec*recall / (prec + recall);
			String threshResults = String.format("thresh: %.3f\n---------------\n prec: %.3f recall: %.3f f1: %.3f\n", fraction, prec, recall, f1);
			result.append(threshResults + "\n"); 
			Execution.putOutput(String.format("Results for thresh %.3f",fraction), threshResults);
			if (f1 > bestF1) {
				bestF1 = f1;
				bestResult = threshResults;
			}
		}
		Execution.putOutput("bestF1", bestF1);
		result.append(String.format("best result\n--------------\n") + bestResult);		
		return result.toString();
	}
	//
	//	public static Pair<Double,String> gradeDictionary(Collection<String> domain, Collection<String> codomain, Collection<String> seedDomain, Collection<String> seedCodomain, CounterMap<String,String> guess, TranslationLexicon goldLexicon, int gradeKMappings, double thresh, boolean verbose) {
	//		int correct = 0;
	//		int seedCorrect = 0;
	//
	//		int seedPossible = seedDomain.size();
	//		int possible = goldLexicon.getPreimage().size() - seedPossible;
	//		int seedGuessed = 0;
	//		int guessed = 0;
	//
	//		StringBuffer seedCorrects = new StringBuffer();
	//		StringBuffer corrects = new StringBuffer();
	//		for (String domainWord : domain) {
	//			PriorityQueue<String> codomainQueue = guess.getCounter(domainWord).asPriorityQueue();
	//			Set<String> goldImage = goldLexicon.getImage(domainWord);
	//			boolean madeGuess = false;
	//			for (int i = 0; i < gradeKMappings; i++) {
	//				if (!codomainQueue.hasNext()) break;
	//				double sim = codomainQueue.getPriority();
	//				if (sim < thresh) break;
	//				madeGuess = true;
	//				String codomainWord = codomainQueue.next();
	//				if (goldImage != null && codomainWord != null && goldImage.contains(codomainWord)) {
	//					if (seedDomain.contains(domainWord)) {
	//						seedCorrect++;
	////						if (verbose)
	////						seedCorrects.append("*seed correct: sim=" + sim + " " + domainWord + " -> " + codomainWord +"\n");
	//					} else {
	//						correct++;
	//						if (verbose)
	//							corrects.append("*correct: sim=" + sim + " " + domainWord + " -> " + codomainWord + "\n");
	//					}
	//					break;
	//				} else {
	//					if (seedDomain.contains(domainWord)) {
	////						if (verbose)
	////						seedCorrects.append("*seed wrong: sim=" + sim + " " + domainWord + " -> " + codomainWord +"\n");
	//					} else {
	//						if (verbose)
	//							corrects.append("*wrong: sim=" + sim + " " + domainWord + " -> " + codomainWord + "\n");
	//					}
	//				}
	//			}
	//			if (madeGuess) {
	//				if (seedDomain.contains(domainWord))
	//					seedGuessed++;
	//				else
	//					guessed++;
	//			}
	//		}
	//
	//		double prec = (double) correct / (double) guessed;
	//		double recall = (double) correct / (double) possible;
	//		double fmeas = 2.0*prec*recall / (prec + recall);
	//		double sprec = (double) seedCorrect / (double) seedGuessed;
	//		double srecall = (double) seedCorrect / (double) seedPossible;
	//		double sfmeas = 2.0*sprec*srecall / (sprec + srecall);
	//
	//		String prefix = seedCorrects.toString() + corrects.toString();
	//		String fmtStr = "Seed possible: %d, Seed guessed: %d, Seed correct: %d, Seed recall: %.3f, Seed prec: %.3f, Seed FMeas: %.3f\nTest Possible: %d, Test Guessed: %d, Test Correct: %d, Test Recall: %.3f, Test Prec: %.3f, Test FMeas: %.3f\n";
	//		String format = prefix + String.format(fmtStr, seedPossible, seedGuessed, seedCorrect, srecall, sprec, sfmeas, possible, guessed, correct, recall, prec, fmeas);
	//		return Pair.newPair(fmeas, format);
	//	}
}
