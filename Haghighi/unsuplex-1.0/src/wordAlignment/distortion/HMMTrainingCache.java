package wordAlignment.distortion;

import java.util.HashMap;
import java.util.Map;

import edu.berkeley.nlp.mt.SentencePair;
import wordAlignment.Model;
import wordAlignment.TrainingCache;

/**
 * Stores shared trellises during a training iteration.
 */
public class HMMTrainingCache extends TrainingCache {

	private int maxCacheSize;

	public HMMTrainingCache(int maxCacheSize) {
		super();
		this.maxCacheSize = maxCacheSize;
	}

	public WATrellis getTrellis(WAState.Factory factory, int I, Model<FirstOrderModel> params, SentencePair pair) {
    WATrellis trellis = trellisCache.get(I);
    if(trellis == null) {
      trellis = new WATrellis(factory, I, params, pair);
      if(I <= maxCacheSize)
        trellisCache.put(I, trellis);
    }
    return trellis;
  }

  public void clear() { trellisCache.clear(); }

  Map<Integer, WATrellis> trellisCache = new HashMap<Integer, WATrellis>();
}
