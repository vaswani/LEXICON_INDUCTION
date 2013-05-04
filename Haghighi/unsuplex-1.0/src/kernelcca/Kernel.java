package kernelcca;

import Jama.Matrix;
import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public interface Kernel<T> {
	public double dot(T x, T y);
}

class LinearKernel implements Kernel<double[]> {
  public double dot(double[] x, double[] y) {
    return ListUtils.dot(x, y);
  }
}


class GaussianKernel implements Kernel<double[]> {
  double var;
  public GaussianKernel(double var) {
    this.var = var;
  }

  public double dot(double[] x, double[] y) {
    return Math.exp(-NumUtils.l2DistSquared(x, y)/(2*var));
  }
}

class NullKernel<T> implements Kernel<T> {
  public double dot(T x, T y) { return 0; }
}

/**
 * Feature space is set of all contiguous substrings.
 */
class AllSubstringKernel implements Kernel<String> {
  public double dot(String x, String y) {
    // SLOW
    int count = 0;
    for(int i = 0; i < x.length(); i++)
      for(int j = 0; j < y.length(); j++)
        for(int k = 0; i+k < x.length() && j+k < y.length() && x.charAt(i) == y.charAt(j); k++)
          count++;
    return count;
  }
}
