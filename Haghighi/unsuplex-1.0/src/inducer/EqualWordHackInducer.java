package inducer;

import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.Pair;

public class EqualWordHackInducer implements DictionaryInducer {
	
	DictionaryInducer base ;
	

	public EqualWordHackInducer(DictionaryInducer base) {
		super();
		this.base = base;
	}

	public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
		double[][] matching = base.getMatchingMatrix(domWords, codomWords);
		for (int i=0; i < domWords.size(); ++i) { 
			for (int j=0; j < codomWords.size(); ++j) {
				if (domWords.getObject(i).equals(codomWords.getObject(j))) {
					matching[i][j] += 1.0;
					matching[i][j] *= 100.0;
				}
			}
		}
		return matching;
	}

	public Pair<double[][], double[][]> getRepresentations(
			Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException();
	}

	public void setSeedMapping(CounterMap<String, String> seedMapping) {
		// TODO Auto-generated method stub
		base.setSeedMapping(seedMapping);
	}

	public void setWords(Indexer<String> domWords, Indexer<String> codomWords) {
		// TODO Auto-generated method stub
		base.setWords(domWords, codomWords);
	}

}
