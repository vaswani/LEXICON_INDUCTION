package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

public class MultivariateGaussianUnitModel implements UnitModel {
  private Matrix Cinv; // Inverse covariance matrix
  final double epsilon = 0.01;

  public void learn(List<double[]> points) {
    int D = points.get(0).length;
    Cinv = new Matrix(D, D);
    for(double[] x : points) {
      for(int i = 0; i < D; i++)
        for(int j = 0; j < D; j++)
          Cinv.set(i, j, x[i]*x[j]);
    }
    for(int i = 0; i < D; i++)
      Cinv.set(i, i, epsilon);
    Cinv = Cinv.inverse();
  }

  public double getScore(double[] x) {
    return MatrixUtils.transformedNormSquared(x, Cinv);
  }
}
