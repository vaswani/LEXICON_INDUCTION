package features;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FeatureManager {
	
	private Map<Feature, Feature> featureInterner = new HashMap<Feature, Feature>();
	private List<Feature> featureList = new ArrayList<Feature>();
	private boolean locked = false;
	
	public void lock() {
		this.locked = true;
	}
	
	public Feature getFeatrue(String pred, String val) {
		return getFeature(pred+"="+val);
	}
	
	public Feature getFeature(int index) {
		return featureList.get(index);
	}
	
	public boolean hasFeature(String val) {
		Feature feat = new Feature(val, -1);
		Feature canonicalFeat = featureInterner.get(feat);
		return canonicalFeat !=  null;
	}
	
	public Feature getFeature(String val) {
		Feature feat = new Feature(val, -1);
		Feature canonicalFeat = featureInterner.get(feat);
		if (canonicalFeat == null) {
			assert !locked;
			feat = new Feature(feat.toString(), featureInterner.size());
			featureInterner.put(feat, feat);
			featureList.add(feat);
			canonicalFeat = feat;
		}
		return canonicalFeat;
	}
	
	public int getNumFeatures() {
		return featureInterner.size();
	}
	
}
