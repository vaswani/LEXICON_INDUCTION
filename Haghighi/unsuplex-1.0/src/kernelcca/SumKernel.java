package kernelcca;

import java.util.ArrayList;
import java.util.List;

public class SumKernel<T> implements Kernel<T> {
	
	private List<Kernel<T>> kernels = new ArrayList<Kernel<T>>();
	private List<Double> weights = new ArrayList<Double>(); 
	
	public SumKernel(Kernel<T>...initKernels) {
		for (Kernel<T> k: initKernels) {
			addKernel(k, 1.0);
		}
	}
	
	public void addKernel(Kernel<T> k, double w) {
		if (w != 0.0) {
			kernels.add(k);
			weights.add(w);
		}
	}
	
	public void setWeights(List<Double> weights) {
		assert weights.size() == kernels.size();
		this.weights = weights;
	}

	public double dot(T x, T y) {
		double sum = 0.0;
		for (int i=0; i<kernels.size(); i++) {
			if (weights.get(i) != 0.0) {
				sum += kernels.get(i).dot(x,y) * weights.get(i);
			}			
		}
//		sum += 1.0e-4;
//		for (Kernel<T> k: kernels) {
//			sum += k.dot(x,y);
//		}
		return sum;
	}

}
