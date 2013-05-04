package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

public class ConstantGaussianUnitModel implements UnitModel {
  public void learn(List<double[]> points) { }

  public double getScore(double[] x) {
    return -NumUtils.l2NormSquared(x);
  }
}
