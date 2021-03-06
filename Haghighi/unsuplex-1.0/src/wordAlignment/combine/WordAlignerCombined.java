package wordAlignment.combine;

import java.util.List;

import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import wordAlignment.EMWordAligner;
import wordAlignment.WordAligner;
import fig.basic.ListUtils;

/**
 * A foundation for aligners that combine two directional alignments
 */
public abstract class WordAlignerCombined extends WordAligner {
	WordAligner wa1, wa2;
	boolean usePosteriorDecodingFlag;
	double posteriorDecodingThreshold;

	public WordAlignerCombined(WordAligner wa1, WordAligner wa2) {
		this.wa1 = wa1;
		this.wa2 = wa2;
		usePosteriorDecodingFlag = EMWordAligner.usePosteriorDecoding;
		posteriorDecodingThreshold = EMWordAligner.posteriorDecodingThreshold;
	}

	public Alignment alignSentencePair(SentencePair sp) {
		return alignSentencePairReturnAll(sp).get(2);
	}

	public List<Alignment> alignSentencePairReturnAll(SentencePair sentencePair) {
		Alignment a1 = wa1.alignSentencePair(sentencePair);
		Alignment a2 = wa2.alignSentencePair(sentencePair);
		Alignment a3 = combineAlignments(a1, a2, sentencePair);
		return ListUtils.newList(a1, a2, a3);
	}

	public void setThreshold(int threshold) {
		posteriorDecodingThreshold = threshold;
	}

	abstract Alignment combineAlignments(Alignment a1, Alignment a2,
			SentencePair sentencePair);
}
