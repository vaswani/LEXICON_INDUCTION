package kernelcca;

import java.util.HashMap;
import java.util.Map;

/**
 * 
 * @author aria42
 *
 * @param <T>
 */
public class CachingVectorRepresenter<T> implements VectorRepresenter<T> {

	VectorRepresenter<T> baseRepn;
	Map<T, double[]> cache = new HashMap<T, double[]>();
	
	
	public CachingVectorRepresenter(VectorRepresenter<T> baseRepn) {
		this.baseRepn = baseRepn;
	}

	public int getD() {
		return baseRepn.getD();
	}

	public double[] getRepn(int i) {
		return baseRepn.getRepn(i);
	}

	public double[] getRepn(T x) {
		double[] repn = cache.get(x);
		if (repn != null) { return repn; }
		repn = baseRepn.getRepn(x);
		cache.put(x, repn);
		return repn;
	}

}