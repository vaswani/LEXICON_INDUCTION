package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

class CenteringVectorRepresenter implements VectorRepresenter<double[]> {
  private double[] mean;

  public CenteringVectorRepresenter(List<double[]> points) {
    this.mean = new double[points.get(0).length];
    for(double[] x : points)
      ListUtils.incr(mean, 1.0, x);
    ListUtils.multMut(mean, 1.0/points.size());
  }
  public int getD() { return mean.length; }
  public double[] getRepn(int i) { throw Exceptions.bad; }
  public double[] getRepn(double[] x) { return ListUtils.sub(x, mean); }
}
