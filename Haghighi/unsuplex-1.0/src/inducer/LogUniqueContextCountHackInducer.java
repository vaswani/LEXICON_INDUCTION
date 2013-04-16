package inducer;

import inducer.LogUniqueContextCountInducer.UniqueContextCounter;

import java.util.List;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import features.ContextKernelFeatureExtractor.ContextType;
import fig.basic.Indexer;
import fig.basic.Pair;

public class LogUniqueContextCountHackInducer implements DictionaryInducer {

	DictionaryInducer inducer ;
	UniqueContextCounter domCounts, codomCounts; 
	
	public LogUniqueContextCountHackInducer(DictionaryInducer inducer, Pair<List<List<String>>,List<List<String>>> corpora, Pair<Indexer<String>,Indexer<String>> words, int windowSize, ContextType contextType) {
		domCounts = new UniqueContextCounter(corpora.getFirst(), words.getFirst(), windowSize, contextType);
		codomCounts = new UniqueContextCounter(corpora.getSecond(), words.getSecond(), windowSize, contextType);
		this.inducer = inducer;
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		double[][] matching = inducer.getMatchingMatrix(domWords, codomWords);
		for (int i=0; i < matching.length;++i) {
			double domLogUnique = Math.log(domCounts.getUniqueContextCountRatio(domWords.getObject(i)));
			for  (int j=0; j < matching[i].length; ++j) {
				double codomLogUnique = Math.log(codomCounts.getUniqueContextCountRatio(codomWords.getObject(j)));
				double diff = Math.abs(domLogUnique-codomLogUnique);		
				matching[i][j] *= Math.exp(-0.1*diff);
			}
		}
		return matching;
	}

	public Pair<double[][], double[][]> getRepresentations(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException();
	}

	public void setSeedMapping(CounterMap<String, String> seedMapping) {
		inducer.setSeedMapping(seedMapping);		
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		
	}
	
}
