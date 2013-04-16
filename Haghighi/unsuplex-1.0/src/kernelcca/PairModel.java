package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

/**
 * Models a pair (x1, x2), where x1 and x2 are real vectors.
 */
public interface PairModel {
  // Train the model with the points.
  public void learn(List<Pair<double[],double[]>> points);

  // Get the score (e.g., log probability or -distance) of a new test point
  public double getScore(double[] x1, double[] x2);
}
