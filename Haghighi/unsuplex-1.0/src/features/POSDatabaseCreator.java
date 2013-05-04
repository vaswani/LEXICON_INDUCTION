package features;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;
import edu.berkeley.nlp.treebank.PennTreebankReader;
import edu.berkeley.nlp.treebank.TreebankFetcher;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

public class POSDatabaseCreator {

	CounterMap<String, String> wordPOSCounts = new CounterMap<String, String>();

	public POSDatabaseCreator(Iterator<Tree<String>> treeIt) {
		while (treeIt.hasNext()) {
			Tree<String> t = treeIt.next();
			List<String> posTags = t.getPreTerminalYield();
			List<String> words = t.getTerminalYield();
			assert posTags.size() == words.size() ;
			for (int i=0; i < words.size(); ++i) {
				wordPOSCounts.incrementCount(words.get(i).toLowerCase(), posTags.get(i).toLowerCase(), 1.0);
			}
		}
		System.out.printf("Read %d words\n", wordPOSCounts.size());
	}

	public void writeFile(String outpath, double minCounts) {
		try {
			int numEntries = 0;
			BufferedWriter writer = new BufferedWriter(new FileWriter(outpath));
			for (String word: wordPOSCounts.keySet()) {				
				Counter<String> tagCounts = wordPOSCounts.getCounter(word);				
				String argMax = tagCounts.argMax();
				if (tagCounts.getCount(argMax) >= minCounts) {
					writer.write(word + "\t" + argMax + "\n");
					numEntries++;
				}							
			}
			writer.close();
			System.out.printf("Wrote %d entries to %s\n",numEntries,outpath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.exit(0);
		}				
	}
	
	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		String inpath = args[0];
		String outpath = args[1];
		double thresh = 2.0;				
//		FileInputStream fis = new FileInputStream(inpath);
//		InputStreamReader in = new InputStreamReader(fis, "utf8"); 
		Iterator<Tree<String>> it = new Trees.PennTreeReader(new FileReader(inpath));
		POSDatabaseCreator dbCreator = new POSDatabaseCreator(it);
		dbCreator.writeFile(outpath, thresh);
	}
	
}
