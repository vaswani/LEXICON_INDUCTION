/**
 * 
 */
package kernelcca;

import java.util.IdentityHashMap;
import java.util.Map;

import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;

public class StringKernel implements Kernel<String> {
	int k = 3; // maximum length
	double[] alphaTable;
//	CounterMap<String, String> memoize = new CounterMap<String, String>();
//	double p = 0.2;
//	Random rand = new Random();
	Map<String, Double> noramlizerCache = new IdentityHashMap<String, Double>();

	public StringKernel() { this(0.9); }
	public StringKernel(double alpha) {
		alphaTable = new double[k+1];
		for (int i=0; i < k+1; ++i) {
			alphaTable[i] = Math.pow(alpha, i);
		}			
	}

	public double lenSquared(String x) {
//		Double cache = noramlizerCache.get(x);
//		if (cache != null) {
//		return cache;
//		}

		x = "#" + x + "#";

		int n = x.length();
		double sum = 0.0;
		char[] xChars = x.toCharArray();

		for (int i=0; i < n; ++i) {
			for (int j=0; j < n; ++j) {
				for (int len=1; len <= k && i+len <= n && j+len <= n; ++len) {
					if (xChars[i+len-1] == xChars[j+len-1]) {
						sum += alphaTable[len];
					} else {
						break;
					}
				}
			}
		}
//		for (int len=1; len <= k; ++len) {

//		String s = x.substring(i,i+len);

//		String t = x.substring(j,j+len);
//		if (s.equals(t)) {
//		double coef = alphaTable[len];
//		sum += coef ;
//		}
//		}
//		}				
//		}
//		noramlizerCache.put(x, sum);
//		LogInfo.logs("normalize: %s %d %.3f\n",x,x.length(),sum);
		return sum;
	}


	public double dot(String x, String y) {

		x = "#" + x + "#";
		y = "#" + y + "#";

//		Counter<String> c = memoize.getCounter(x);
//		if (c.containsKey(y)) {
//		return c.getCount(y);
//		}

		int m = x.length();
		int n = y.length();
		double sum = 0.0;
		char[] xChars = x.toCharArray();
		char[] yChars = y.toCharArray();
		for (int i=0; i < m; ++i) {
			for (int j=0; j < n; ++j) {
				for (int len=1; len <= k && i+len <= m && j+len <= n; ++len) {
					if (xChars[i+len-1] == yChars[j+len-1]) {
						sum += alphaTable[len];
					} else {
						break;
					}
				}
			}
		}
//		for (int len=1; len <= k; ++len) {
//		for (int i=0; i+len < m; ++i) {
//		String s = x.substring(i,i+len);
//		for (int j=0; j+len < n; ++j) {					
//		String t = y.substring(j,j+len);
//		if (s.equals(t)) {
//		double coef = alphaTable[len];
//		sum += coef ;
//		}
//		}
//		}				
//		}

		double xLen = Math.sqrt(lenSquared(x));
		double yLen = Math.sqrt(lenSquared(y));
		assert xLen > 0.0 && yLen > 0.0;
		sum /= (xLen * yLen);

//		assert Math.abs(sum) <= 1.0; // cauchy-schwartz

//		if (rand.nextDouble() < p) {
//		memoize.setCount(x, y, sum);
//		}

		return sum;
	}

}
