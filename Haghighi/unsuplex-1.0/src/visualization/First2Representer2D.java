package visualization;

import java.util.ArrayList;
import java.util.List;

import fig.basic.Pair;

public class First2Representer2D implements Representer2D {

	public List<Pair<Double, Double>> get2DRepresentations(double[][] reps) {
		List<Pair<Double,Double>> reps2D = new ArrayList<Pair<Double,Double>>();
		for (double[] rep : reps) {
			reps2D.add(Pair.newPair(rep[0], rep[1]));
		}
		return reps2D;
	}

}
