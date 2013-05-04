package inducer;

import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.Triple;
import features.ContextKernelFeatureExtractor;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;

import inducer.DictionaryInducerTester.Options;
import io.POSTagPrefixes;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import kernelcca.CachingKernel;
import kernelcca.Kernel;
import kernelcca.SumKernel;

public class BestKernel {
	
	private static Counter<String> getWordScorer(Indexer<String> indexer) {
		Counter<String> counts = new Counter<String>();
		for (int i=0; i < indexer.size(); ++i) {
			counts.setCount(indexer.getObject(i), Math.log(1+i));
		}
		return counts;
	}
	
	public static Triple<Kernel<String>,Kernel<String>,Kernel<String>>  getBestKernel(Pair<Indexer<String>,Indexer<String>> words, NewBitext bitext, Options opts) {
		double p = 0.2;
		Kernel<String> strKernel = new CachingKernel<String>(new kernelcca.StringKernel(),p);
//		Kernel<String> sharedKernel = opts.useSharedStringKernel ? new StringKernel() : null;

		if (opts.contextWeight == 0.0) {
			return Triple.makeTriple((Kernel<String>) new CachingKernel<String>(strKernel, p), (Kernel<String>)  strKernel, null);
		}
		
		// get context kernels
		Set<POSTagPrefixes.POSTag> contextTags = POSTagPrefixes.getTagSet(opts.contextPOS);
		Set<POSTagPrefixes.POSTag> wordTags = POSTagPrefixes.getTagSet(opts.wordsPOS);
		//		opts.bitextProcessor.setPOSTags(contextTags);
		Pair<Indexer<String>,Indexer<String>> featWords = bitext.getMostCommonWordsInOrder(opts.numContextWords, contextTags);
		Pair<Indexer<String>,Indexer<String>> extractWords = bitext.getMostCommonWordsInOrder(opts.seedSize + opts.numWords, wordTags);
		
		int numOriginal = bitext.getBaseCorpus().getFirst().size();
		Counter<String> domScorer = getWordScorer(extractWords.getFirst());
		Counter<String> codomScorer = getWordScorer(extractWords.getSecond());
		
		Pair<? extends Iterator<List<String>>, ? extends Iterator<List<String>>> reducedCorpora = bitext.getReducedFullCorpus(contextTags,domScorer, codomScorer);
		Iterator<List<String>> domainCorpus = reducedCorpora.getFirst();
		Iterator<List<String>> codomainCorpus = reducedCorpora.getSecond();
		
		Kernel<String> domContextKernel = new LinearSparseKernel<String>(new ContextKernelFeatureExtractor(domainCorpus, extractWords.getFirst(), featWords.getFirst(), opts.windowSize, opts.contextType,numOriginal));
		Kernel<String> codomContextKernel = new LinearSparseKernel<String>(new ContextKernelFeatureExtractor(codomainCorpus, extractWords.getSecond(), featWords.getSecond(), opts.windowSize, opts.contextType,numOriginal));
		
		if (opts.orthoWeight == 0.0) {
			return Triple.makeTriple((Kernel<String>) domContextKernel, codomContextKernel, (Kernel<String>)  null);
		}
		
		List<Double> weights = CollectionUtils.makeList(opts.orthoWeight, opts.contextWeight);
		
		SumKernel<String> domSumKernel = new SumKernel<String>(strKernel, domContextKernel);
		domSumKernel.setWeights(weights);
		SumKernel<String> codomSumKernel = new SumKernel<String>(strKernel, codomContextKernel);
		codomSumKernel.setWeights(weights);
						
		return Triple.makeTriple((Kernel<String>) domSumKernel, (Kernel<String>) codomSumKernel, null);
	}
}
