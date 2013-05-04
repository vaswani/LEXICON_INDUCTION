package kernelcca;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

public class CachingKernel<T> implements Kernel<T> {

	private Kernel<T> kernel;
	private CounterMap<T, T> kernelCache = new CounterMap<T, T>();
	private double p ;
	private java.util.Random rand = new java.util.Random();
	
	public CachingKernel(Kernel<T> kernel) {
		this(kernel, 1.0);
	}
	
	public CachingKernel(Kernel<T> kernel, double p) {
		this.kernel = kernel ;
		this.p = p ;
	}
	
	public double dot(T x, T y) {		
		if (kernelCache.containsKey(x)) {
			Counter<T> xCache = kernelCache.getCounter(x);
			if (xCache.containsKey(y)) {
				return xCache.getCount(y);
			}
		}
		
		double d = kernel.dot(x,y);
		if (rand.nextDouble() < p) {
			kernelCache.setCount(x, y, d);
		}		
		return d;
	}

}
