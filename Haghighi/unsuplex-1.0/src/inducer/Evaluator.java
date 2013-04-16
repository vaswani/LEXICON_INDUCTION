package inducer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;


import translex.BilingualLexicon;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Pair;
import fig.exec.Execution;

public class Evaluator  {
	//	Indexer<String> domain;
	//	Indexer<String> codomain ;
	//	CounterMap<String, String> seedMapping ;
	BilingualLexicon goldLexicon;
	int threshLevels;
	boolean verbose;
	public static Options opts = new Options();
	double precisionAtTargetRecall = 0.0;

	public static class Options {
		@Option
		public double evalAlpha = 1.0;
		@Option
		public List<Double>	recallTargets = CollectionUtils.makeList(0.1,0.25,0.33,0.5,0.75);
	}

	public Evaluator(BilingualLexicon goldLexicon, int threshLevels, boolean verbose) {
		this.goldLexicon = goldLexicon;
		this.threshLevels = threshLevels;
		this.verbose = verbose;
	}

	public static class Prediction implements Comparable<Prediction> {
		String srcWord, trgWord;
		double score ;
		boolean correct ;
		public Prediction(String srcWord, String trgWord, double score) {
			super();
			this.srcWord = srcWord;
			this.trgWord = trgWord;
			this.score = score;
		}
		public String toString( ) {
			return String.format("[%s] %s -> %s %.3f",correct ? "correct" : "wrong", srcWord, trgWord, score);
		}
		public int compareTo(Prediction o) {
			if (o.score > this.score) return 1;
			if (o.score == this.score) return 0;
			return -1;
		}		
	}
	
	private static <T extends Comparable<T>> List<T> getSortedList(Collection<T> coll) {
		List<T> lst = new ArrayList<T>(coll);
		Collections.sort(lst);
		return lst;
	}

	public List<Prediction> getSortedAnnotatedPrediction(CounterMap<String, String> guess) {
		List<Prediction> predictions = new ArrayList<Prediction>();		
		for (String dom: getSortedList(guess.keySet())) {
			Counter<String> domGuesses = guess.getCounter(dom);
			for (String codom: getSortedList(domGuesses.keySet())) {
				Prediction prediction = new Prediction(dom,codom,domGuesses.getCount(codom));
				prediction.correct = goldLexicon.containsTranslation(dom, codom); 
				predictions.add(prediction);
			}
		}

		Collections.sort(predictions);
		return predictions;
	}

	public static List<Prediction> getSortedPrediction(CounterMap<String, String> guess) {
		List<Prediction> predictions = new ArrayList<Prediction>();
		for (String dom: getSortedList(guess.keySet())) {
			Counter<String> domGuesses = guess.getCounter(dom);
			for (String codom: getSortedList(domGuesses.keySet())) {
				predictions.add(new Prediction(dom,codom,domGuesses.getCount(codom)));
			}
		}
		Collections.sort(predictions);
		return predictions;
	}

	private static class FMeasure {
		double prec, recall, f1 ;

		public FMeasure(double prec, double recall, double f1) {
			super();
			this.prec = prec;
			this.recall = recall;
			this.f1 = f1;
		}

		public String toString() {
			return String.format("prec: %.3f, recall: %.3f f1: %.3f", prec,recall, f1);
		}
	}

	private FMeasure getFMeasure(List<Prediction> preds, int possible) {
		int correct=0;//, guessed=0;
		for (Prediction p: preds) {				
			if (p.correct) {
				correct++;
			} 
		}
		double prec = correct / (double) preds.size();
		double recall = correct/ (double) possible;
		double f1 = (1+opts.evalAlpha)*prec*recall / (opts.evalAlpha * prec + recall);
		return new FMeasure(prec,recall,f1);
	}
	
	private List<Prediction> filterByLexicon(List<Prediction> preds) {
		List<Prediction> filteredPreds = new ArrayList<Prediction>();
		for (Prediction p: preds) {
			if (goldLexicon.containsTranslation(p.srcWord)) {
				filteredPreds.add(p);
			}
		}
		return filteredPreds;
		
	}

	public String eval(final CounterMap<String, String> guess, Collection<String> domWords, Collection<String> codomWords) {
		StringBuilder result = new StringBuilder();
		List<Prediction> predictions = getSortedAnnotatedPrediction(guess);
//		Set<String> toGuessOn = new HashSet<String>();
//		LogInfo.logs("lexicon size: " + goldLexicon.size());
//		LogInfo.logs("domWords: " + domWords);
//		LogInfo.logs("codomWords: " + domWords);
		int possible = 0;
		
		for (String dom: getSortedList(domWords)) {
			if (!goldLexicon.containsTranslation(dom)) {
				continue;
			}
			Set<String> img = goldLexicon.getImage(dom);
			if (img.isEmpty()) { continue; }
			boolean intersect = false;
			for (String codom: getSortedList(img)) { 
				if (codomWords.contains(codom)) { 
					intersect = true;
				}
			}; 
			if (intersect) { possible++ ; }
		}
//		LogInfo.logs("size of predictions before: " + predictions.size());
		Iterator<Prediction> it = predictions.iterator();
		while (it.hasNext()) {
			Prediction p = it.next();
			if (!goldLexicon.containsTranslation(p.srcWord)) {
				it.remove();
			}		
		}	
//		LogInfo.logs("size of predictions after: " + predictions.size());
		
//		LogInfo.logs("domSize: %d codomSize: %d",domWords.size(), codomWords.size());
//		LogInfo.logs("numPossible %d",possible);
//		LogInfo.logs("To Guess On: " + toGuessOn);
		int numCorrect = 0, numGuess = 0;
		Map<Double, Double> precisionAtRecallMap = new HashMap<Double, Double>();
		for (Prediction p: predictions) {
			assert goldLexicon.containsTranslation(p.srcWord);
			//p.correct = goldLexicon.getImage(p.srcWord).contains(p.trgWord);			
			numGuess++;
			if (p.correct) {
				numCorrect++;
				for (double r: opts.recallTargets) {
					int target = (int) (r * possible);
					if (numCorrect == target) {
						double precision = ((double) numCorrect) / ((double) numGuess);
						precisionAtRecallMap.put(r, precision);
					}
				}
			}			
		}

		FMeasure bestFMeasure = new FMeasure(0.0,0.0,0.0); 
		int bestTopN = -1;
		FMeasure evenFMeasure = null;
		double lowestPrecRecallDist = Double.POSITIVE_INFINITY;
		int evenTopN = -1;

		for (int i=0; i < predictions.size(); ++i) {
			List<Prediction> curPreds = predictions.subList(0, i);//filterByLexicon(predictions.subList(0, i));
			FMeasure fMeasure = getFMeasure(curPreds, possible);
			if (fMeasure.f1 > bestFMeasure.f1) {
				bestFMeasure = fMeasure;
				bestTopN = curPreds.size();
			}
			double precRecallDist = Math.abs(fMeasure.prec - fMeasure.recall);
			if (!Double.isNaN(fMeasure.f1) && precRecallDist < lowestPrecRecallDist) {
				evenFMeasure = fMeasure;
				lowestPrecRecallDist = precRecallDist;
				evenTopN = curPreds.size();
			}
		}
		double bestFraction = ((double) (bestTopN)) / ((double) predictions.size());
		String bestResult = String.format("frac: %.3f (%d/%d) result: %s",bestFraction,bestTopN,predictions.size(),bestFMeasure);
		double evenFracton = ((double) (evenTopN)) / ((double) predictions.size());
		String evenResult = String.format("frac: %.3f (%d/%d) result: %s",evenFracton,evenTopN,predictions.size(),evenFMeasure);
		Execution.putOutput("bestF1", bestResult);
		Execution.putOutput("evenF1", evenResult);
		Execution.putOutput("P@R", precisionAtRecallMap);
		result.append(String.format("best F1 result: ") + bestResult + "\n");
		result.append(String.format("even F1 result: ") + evenResult + "\n");
		result.append(String.format("P@R=  %s", precisionAtRecallMap));
		
		return result.toString();
	}

	List<Pair<Double,Double>> getPrecisionRecallPoints(final CounterMap<String, String> guess, Indexer<String> domWords, Indexer<String> codomWords) {
		List<Prediction> predictions =getSortedAnnotatedPrediction(guess);
		int possible = 0;
		for (String dom: domWords) {
			if (!goldLexicon.containsTranslation(dom)) {
				continue;
			}
			Set<String> img = goldLexicon.getImage(dom);

			if (img.isEmpty()) { continue; }
			boolean intersect = false;
			for (String codom: img) { 
				if (codomWords.contains(codom)) { intersect = true; }
			}; 
			if (intersect) { possible++ ; }
		}
		for (Prediction p: predictions) {
			p.correct = goldLexicon.getImage(p.srcWord).contains(p.trgWord);						
		}

		List<Pair<Double, Double>> pairs = new ArrayList<Pair<Double,Double>>();
		for (int i=0; i < predictions.size(); ++i) {
			List<Prediction> curPreds = predictions.subList(0, i);
			FMeasure fMeasure = getFMeasure(curPreds, possible);
			pairs.add(Pair.newPair(fMeasure.prec,fMeasure.recall));
		}
		return pairs;
	}
	
	
}
