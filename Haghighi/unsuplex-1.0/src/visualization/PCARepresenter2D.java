package visualization;

import java.util.ArrayList;
import java.util.List;

import kernelcca.Kernel;
import kernelcca.KernelPCA;
import kernelcca.VectorRepresenter;

import fig.basic.ListUtils;
import fig.basic.Pair;

public class PCARepresenter2D implements Representer2D {
	
	private VectorRepresenter<double[]> pcaRepresentor;
	
	public PCARepresenter2D(Pair<double[][],double[][]> reps) {
		// set up pca
		KernelPCA<double[]> pca = new KernelPCA<double[]>();
		List<double[]> data = new ArrayList<double[]>();
		for (int i=0; i < reps.getFirst().length; ++i) {
			data.add(reps.getFirst()[i]);
		}
		for (int i=0; i < reps.getSecond().length; ++i) {
			data.add(reps.getFirst()[i]);
		}
		pca.setData(data, new Kernel<double[]>() {
			public double dot(double[] x, double[] y) {
				return ListUtils.dot(x, y);
			}});
		pcaRepresentor = pca.solve();
	}

	public List<Pair<Double, Double>> get2DRepresentations(double[][] reps) {
		// get new reps
		List<Pair<Double,Double>> reps2D = new ArrayList<Pair<Double,Double>>();
		for (int i=0; i < reps.length; ++i) {
			double[] input = reps[i];
			double[] output = pcaRepresentor.getRepn(input);
			reps2D.add(Pair.newPair(output[0], output[1]));
		}
		return reps2D;
	}

}
