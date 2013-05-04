package inducer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.Collection;
import java.util.HashSet;

import translex.BilingualLexicon;

import edu.berkeley.nlp.util.CounterMap;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

public class EvaluatorTester implements Runnable {
	
	public static class Options {
		@Option(required=true)
		public String proposedLexicon ;		
		@Option(required=true)
		public String evalLexicon ;
		@Option
		public boolean ignoreCase = true;
	}
	
	public static Options opts = new Options();
		
	Collection<String> domWords = new HashSet<String>();
	Collection<String> codomWords = new HashSet<String>();
	CounterMap<String, String> guess = new CounterMap<String, String>();
	BilingualLexicon bilingualLexicon = new BilingualLexicon();
	
	private void readGuess() {
		try {
			FileInputStream fi = new FileInputStream(opts.proposedLexicon);
			Reader ir = new InputStreamReader(fi, "UTF8"); // hardcoded
			BufferedReader br = new BufferedReader(ir);
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				if (opts.ignoreCase) {
					line = line.toLowerCase();					
				}
				String[] fields = line.split("\\s+");
				String dom = fields[0];
				String codom = fields[1];
				double conf = Double.parseDouble(fields[2]);
				domWords.add(dom);
				codomWords.add(codom);
				guess.setCount(dom, codom, conf);
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	private void readEval() {
		try {
			FileInputStream fi = new FileInputStream(opts.evalLexicon);
			Reader ir = new InputStreamReader(fi, "UTF8"); // hardcoded
			BufferedReader br = new BufferedReader(ir);
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				if (opts.ignoreCase) {
					line = line.toLowerCase();					
				}
				try {
				String[] fields = line.split("\\s+");
//				assert fields.length == 2;
				String domWord = fields[0];
				String codomWord = fields[1];
				if (domWords.contains(domWord) && codomWords.contains(codomWord)) {
					bilingualLexicon.addTranslation(domWord, codomWord);	
				} 
				} catch (Exception e) { }			
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}		
	}
	
	public void run() {		
		readGuess();
//		NewBitext bitext = new NewBitext(DictionaryInducerTester.opts.basePath,DictionaryInducerTester.opts.bitextCorpusExtractor,DictionaryInducerTester.opts.maxSentences,DictionaryInducerTester.opts.maxAdditionalSents);
//		bilingualLexicon = bitext.getLexicon();
		readEval();
		Evaluator evaluator = new Evaluator(bilingualLexicon,0,false);		
		LogInfo.logs(evaluator.eval(guess, domWords, codomWords));
	}
	
	public static void main(String[] args) {
		Execution.run(args, new EvaluatorTester(), EvaluatorTester.opts, "evalOptions",Evaluator.opts);
	}
	
	
}
