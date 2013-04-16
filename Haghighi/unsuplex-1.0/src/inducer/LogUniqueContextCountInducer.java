package inducer;

import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.util.*;
import features.ContextKernelFeatureExtractor.ContextType;
import fig.basic.Indexer;
import fig.basic.Pair;

public class LogUniqueContextCountInducer implements DictionaryInducer {

	public static class UniqueContextCounter {

		public CounterMap<String,Object> contextCounts = new CounterMap<String,Object>();

		public double getUniqueContextCountRatio(String word) {
			return ((double) contextCounts.getCounter(word).keySet().size() + 1.0) / ((double) contextCounts.getCounter(word).totalCount() + 1.0);
		}
		
		public UniqueContextCounter(List<List<String>> docs, Indexer<String> wordsToExtract, int windowSize, ContextType contextType) {

			for (List<String> doc: docs) {
				for (int i=0; i < doc.size(); ++i) {
					String word = doc.get(i);
					
					// only look at words in domain/codomain
					if (!wordsToExtract.contains(word)) {
						continue;
					}

					int startIndex = Math.max(i-windowSize,0);
					int stopIndex = Math.min(i+windowSize+1, doc.size());
					Object context = null;

					switch (contextType) {
					case POSITIONAL:
						context = doc.subList(startIndex, stopIndex);
						break;
					case DIRECTIONAL:
						context = Pair.newPair(new HashSet<String>(doc.subList(startIndex,i)), new HashSet<String>(doc.subList(i+1,stopIndex)));
						break;
					case UNDISTINGUISHED:							
						context = new HashSet<String>(doc.subList(startIndex,stopIndex));
						break;
					default:
						throw new Error("Bad context type.");
					}

					contextCounts.incrementCount(word, context, 1.0);
				}
			}

		}
	}
	
	private UniqueContextCounter domCounts; 
	private UniqueContextCounter codomCounts; 
	
	public LogUniqueContextCountInducer(Pair<List<List<String>>,List<List<String>>> corpora, Pair<Indexer<String>,Indexer<String>> words, int windowSize, ContextType contextType) {
		domCounts = new UniqueContextCounter(corpora.getFirst(), words.getFirst(), windowSize, contextType);
		codomCounts = new UniqueContextCounter(corpora.getSecond(), words.getSecond(), windowSize, contextType);
	}
	
	private double getCountSim(String domWord, String codomWord) {
		double domUnqiue = domCounts.getUniqueContextCountRatio(domWord);
		double codomUnique = codomCounts.getUniqueContextCountRatio(codomWord);
		return -Math.abs(Math.log(domUnqiue) - Math.log(codomUnique));
	}
	

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		int n = domWords.size();
		double[][] simMat = new double[n][n];
		for (int i=0; i < domWords.size(); ++i) {
			for (int j=0; j < codomWords.size(); ++j) {
				double sim = (double)getCountSim(domWords.getObject(i), codomWords.getObject(j));
				simMat[i][j] = sim; 
			}
		}
		return simMat;
	}

	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		throw new UnsupportedOperationException();
	}

	public void setSeedMapping(CounterMap<String, String> seedMapping) {
		// ignored
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		
	}


}
