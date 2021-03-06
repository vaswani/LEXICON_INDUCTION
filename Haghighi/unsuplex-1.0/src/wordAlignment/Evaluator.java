package wordAlignment;

import static fig.basic.LogInfo.end_track;
import static fig.basic.LogInfo.logs;
import static fig.basic.LogInfo.logss;
import static fig.basic.LogInfo.track;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.SentencePairReader.PairDepot;
import fig.basic.IOUtils;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OutputOrderedMap;
import fig.basic.StrUtils;
import fig.basic.String2DoubleMap;
import fig.exec.Execution;

/**
 * The evaluator can both test the a model and search for its optimal
 * posterior threshold assuming posterior decoding.
 */
public class Evaluator {
	@Option(gloss = "Evaluate using line search")
	public static boolean searchForThreshold = false;
	@Option(gloss = "Sets the number of intervals for posterior threshold line search")
	public static int thresholdIntervals = 20;
	@Option(gloss = "Save object files for proposed alignments (large files)")
	public static boolean saveAlignmentObjects = false;

	List<SentencePair> testSentencePairs;
	String2DoubleMap dictionary;

	public Evaluator(PairDepot testSentencePairs) {
		this(testSentencePairs, null);
	}

	public Evaluator(PairDepot testSentencePairs, String2DoubleMap dictionary) {
		this.testSentencePairs = testSentencePairs.asList();
		this.dictionary = dictionary;
	}

	public Performance test(WordAligner wordAligner, boolean output) {
		return test(wordAligner, output, searchForThreshold);
	}

	public Performance test(WordAligner wordAligner, boolean output, boolean evalPRTradeoff) {
		track("Testing " + wordAligner.getName());

		// Main computation: align sentences!
		Map<Integer, Alignment> proposed = wordAligner.alignSentencePairs(testSentencePairs);

		// Evaluate performance given fixed decoding parameters
		Performance mainPerf = eval(testSentencePairs, proposed);
		mainPerf.bestAER = mainPerf.aer;
		mainPerf.bestThreshold = EMWordAligner.posteriorDecodingThreshold;

		// Do precision/recall tradeoff: only meaningful if we were using posterior decoding
		if (evalPRTradeoff) {
			track("Evaluate precision/recall tradeoff");
			// Get an entire curve
			OutputOrderedMap<Double, String> postMap = new OutputOrderedMap<Double, String>(
					Execution.getFile(wordAligner.modelPrefix + ".PRTradeoff"));
			for (int i = 0; i < thresholdIntervals; i++) {
				double threshold = 1.0 * i / thresholdIntervals;
				Map<Integer, Alignment> thresholded = thresholdAlignments(wordAligner, proposed,
						threshold);
				Performance perf = eval(testSentencePairs, thresholded);
				postMap.put(threshold, perf.simpleString());
				logs("Threshold = %f; AER = %f", threshold, perf.aer);
				if (perf.aer < mainPerf.bestAER) {
					mainPerf.bestAER = perf.aer;
					mainPerf.bestThreshold = threshold;
				}
			}
			logss("Best threshold = %f, AER = %f", mainPerf.bestThreshold, mainPerf.bestAER);
			end_track();
		}

		// Output alignments
		track("Output alignments");
		String file = Execution.getFile(wordAligner.modelPrefix);
		if (output && file != null) {
			//			proposedAlignments = thresholdAlignments(wordAligner, proposedAlignments,
			//					bestThreshold);
			AlignmentsInfo ainfo = new AlignmentsInfo(wordAligner.getName(), testSentencePairs,
					proposed, dictionary);
			if (saveAlignmentObjects) ainfo.writeBinary(file + ".alignOutput.bin");
			ainfo.writeText(file + ".alignOutput.txt");
			ainfo.writeGIZA(file + ".alignOutput.A3");
			ainfo.writePharaoh(file + ".alignOutput.align");
		}
		end_track();

		mainPerf.dump();

		end_track();
		return mainPerf;
	}

	private Map<Integer, Alignment> thresholdAlignments(WordAligner wa,
			Map<Integer, Alignment> proposedAlignments, double threshold) {
		Map<Integer, Alignment> map = new HashMap<Integer, Alignment>();
		for (Integer i : proposedAlignments.keySet()) {
			Alignment al = proposedAlignments.get(i);
			map.put(i, wa.thresholdAlignment(al, threshold));
		}
		return map;
	}

	// Evaluate the proposed alignments against the reference alignments.
	public static Performance eval(List<SentencePair> testSentencePairs,
			Map<Integer, Alignment> proposedAlignments) {
		Performance perf = new Performance();

		//int idx = 0;
		for (SentencePair sentencePair : testSentencePairs) {
			//logs("Sentence %d/%d", idx++, testSentencePairs.size());

			int I = sentencePair.I();
			int J = sentencePair.J();

			Alignment proposedAlignment = proposedAlignments.get(sentencePair.getSentenceID());
			Alignment referenceAlignment = sentencePair.getAlignment();

			// Silently ignore alignments that aren't there
			if (proposedAlignments == null || referenceAlignment == null)
				LogInfo.error("Missing alignment during evaluation.  ID: "
						+ sentencePair.getSentenceID());

			boolean[] hit1 = new boolean[I];
			boolean[] hit2 = new boolean[J];

			for (int j = 0; j < J; j++) {
				for (int i = 0; i < I; i++) {
					boolean proposed = proposedAlignment.containsSureAlignment(i, j);
					boolean sure = referenceAlignment.containsSureAlignment(i, j);
					boolean possible = referenceAlignment.containsPossibleAlignment(i, j);
					double strength = proposedAlignment.getStrength(i, j);

					perf.addPoint(proposed, sure, possible, strength);
					if (proposed) hit1[i] = hit2[j] = true;
				}
			}

			for (int i = 0; i < I; i++)
				if (!hit1[i]) perf.numNull1++;
			for (int j = 0; j < J; j++)
				if (!hit2[j]) perf.numNull2++;
		}

		perf.computeFromCounts();
		return perf;
	}

	/**
	 * This produces two sets of alignments
	 * which look like they were produced by GIZA and one that looks like it
	 * was produced by the Pharaoh training scripts. These alignments will be used
	 * to construct phrases. The output should have the property that the
	 * intersection is the output of the intersected model, and the union is the
	 * union of the two models.
	 */
	static void writeAlignments(PairDepot pairs, WordAligner wa, String prefix) {
		track("Writing directional and union alignments for %d sentences", pairs.size());

		String enSuff = Main.englishSuffix;
		String frSuff = Main.foreignSuffix;
		//		String e2fName = "training." + enSuff + "2" + frSuff + ".A3";
		//		String f2eName = "training." + frSuff + "2" + enSuff + ".A3";
		String unionE2fName = prefix + "." + enSuff + "-" + frSuff + ".A3";
		String unionF2eName = prefix + "." + frSuff + "-" + enSuff + ".A3";
		String unionName = prefix + "." + enSuff + "-" + frSuff + ".align";
		String eInput = prefix + "." + enSuff + "Input.txt";
		String eTrees = prefix + "." + enSuff + "Trees.txt";
		String fInput = prefix + "." + frSuff + "Input.txt";

		//		PrintWriter efOut = IOUtils.openOutHard(Execution.getFile(e2fName));
		//		PrintWriter feOut = IOUtils.openOutHard(Execution.getFile(f2eName));
		PrintWriter unionE2fOut = IOUtils.openOutHard(Execution.getFile(unionE2fName));
		PrintWriter unionF2eOut = IOUtils.openOutHard(Execution.getFile(unionF2eName));
		PrintWriter unionPharaohOut = IOUtils.openOutHard(Execution.getFile(unionName));
		PrintWriter eInputOut = IOUtils.openOutHard(Execution.getFile(eInput));
		PrintWriter eTreesOut = IOUtils.openOutHard(Execution.getFile(eTrees));
		PrintWriter fInputOut = IOUtils.openOutHard(Execution.getFile(fInput));

		int idx = 0;
		for (SentencePair sp : pairs) {
			logs("Sentence %d/%d", idx, pairs.size());
			idx++;
			//			List<Alignment> a123 = wa.alignSentencePairReturnAll(sp);
			//			Alignment a1 = a123.get(0); // E->F
			//			Alignment a2 = a123.get(1); // F->E
			Alignment a3 = wa.alignSentencePair(sp); // Combined
			//			if (competitiveThresholding) a3 = competitiveThresholding(a3, trainingThreshold);

			//			a1.writeGIZA(efOut, idx);
			//			a2.reverse().writeGIZA(feOut, idx);
			a3.writeGIZA(unionE2fOut, idx);
			a3.reverse().writeGIZA(unionF2eOut, idx);

			eInputOut.println(StrUtils.join(sp.getEnglishWords(), " "));
			fInputOut.println(StrUtils.join(sp.getForeignWords(), " "));
			if (sp.getEnglishTree() != null) eTreesOut.println(sp.getEnglishTree());

			unionPharaohOut.println(a3.outputHard());
		}

		//		efOut.close();
		//		feOut.close();
		unionE2fOut.close();
		unionF2eOut.close();
		unionPharaohOut.close();
		eInputOut.close();
		eTreesOut.close();
		fInputOut.close();
		end_track();
	}

}
