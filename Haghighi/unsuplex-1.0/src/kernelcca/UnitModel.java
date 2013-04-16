package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

/**
 * Models a single real vector x.
 */
public interface UnitModel {
  // Train the model with the points.
  public void learn(List<double[]> points);

  // Get the score (e.g., log probability or -distance) of a new test point
  public double getScore(double[] x);
}
