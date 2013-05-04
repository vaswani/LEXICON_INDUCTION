package inducer;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.Pair;

public class LogCountHackInducer implements DictionaryInducer {

	DictionaryInducer inducer ;
	Counter<String> domCounts, codomCounts; 
	
	public LogCountHackInducer(DictionaryInducer inducer, Counter<String> domCounts, Counter<String> codomCounts) {
		this.inducer = inducer;
		this.domCounts = domCounts;
		this.codomCounts = codomCounts;
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		double[][] matching = inducer.getMatchingMatrix(domWords, codomWords);
		for (int i=0; i < matching.length;++i) {
			double domLogCount = Math.log(domCounts.getCount(domWords.getObject(i)));
			for  (int j=0; j < matching[i].length; ++j) {
				double codomLogCount = Math.log(codomCounts.getCount(codomWords.getObject(j)));
				double diff = Math.abs(domLogCount-codomLogCount);		
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
