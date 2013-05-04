package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

public class MatrixUtils { 
  public static double transformedNormSquared(double[] x, Matrix sigmaInv) {
	assert x.length == sigmaInv.getRowDimension() ; 
    Matrix XT = new Matrix(new double[][] {x});
    Matrix X = XT.transpose();
    return XT.times(sigmaInv).times(X).get(0, 0);
  }

  public static void logMatrix(String s, Matrix m) {
    track(s, true);
    double[][] a = m.getArray();
    for(int i = 0; i < a.length; i++)
      logs(Fmt.D(a[i]));
    end_track();
  }

  // Create a block matrix:
  // [aa ab]
  // [ba bb]
  public static Matrix block(Matrix aa, Matrix ab, Matrix ba, Matrix bb) {
    int na = (aa != null ? aa : ab).getRowDimension();
    int nb = (bb != null ? bb : ab).getColumnDimension();
    int n = na+nb;
    assert aa.getRowDimension() == na && aa.getColumnDimension() == na;
    assert ab.getRowDimension() == na && ab.getColumnDimension() == nb;
    assert ba.getRowDimension() == nb && ba.getColumnDimension() == na;
    assert bb.getRowDimension() == nb && bb.getColumnDimension() == nb;

    Matrix m = new Matrix(n, n);
    if(aa != null) m.setMatrix(0, na-1, 0, na-1, aa);
    if(ab != null) m.setMatrix(0, na-1, na, n-1, ab);
    if(ba != null) m.setMatrix(na, n-1, 0, na-1, ba);
    if(bb != null) m.setMatrix(na, n-1, na, n-1, bb);
    return m;
  }
  
  public static double[] getRow(Matrix m, int i) {
	  double[] a = new double[m.getColumnDimension()];
	  for (int j=0; j < a.length; ++j) {
		  a[j] = m.get(i, j);
	  }
	  return a;
  }

  public static double[] getColumn(Matrix m, int j) {
    double[] a = new double[m.getRowDimension()];
    for(int i = 0; i < a.length; i++)
      a[i] = m.get(i, j);
    return a;
  }

  // Return column j from row [i0, i1)
  public static double[] getColumn(Matrix m, int j, int i0, int i1) {
    double[] a = new double[i1-i0];
    for(int i = 0; i < a.length; i++)
      a[i] = m.get(i0+i, j);
    return a;
  }

  // Normalize v so that v^T A v = 1.
  public static double[] metricNormalizeMut(double[] v, Matrix A) {
    double sum = 0;
    for(int i = 0; i < v.length; i++)
      for(int j = 0; j < v.length; j++)
        sum += v[i] * v[j] * A.get(i, j);
    if(sum > 0)
      ListUtils.multMut(v, 1.0/Math.sqrt(sum));
    return v;
  }

  // ni = number of rows, nj = number of columns
  public static Matrix randOrthogonalMatrix(Random random, int ni, int nj) {
    int n = Math.max(ni, nj);
    Matrix m = randGaussianMatrix(random, n, n);
    return m.qr().getQ().getMatrix(0, ni-1, 0, nj-1);
  }

  // ni = number of rows, nj = number of columns
  public static Matrix randGaussianMatrix(Random random, int ni, int nj) {
    Matrix m = new Matrix(ni, nj);
    for(int i = 0; i < ni; i++)
      for(int j = 0; j < nj; j++)
        m.set(i, j, SampleUtils.sampleGaussian(random));
    return m;
  }

  // Normalize the columns of the matrix
  // (example application: columns are an orthogonal basis)
  public static void normalizeColumns(Matrix m) {
    int ni = m.getRowDimension();
    int nj = m.getColumnDimension();
    for(int j = 0; j < nj; j++) {
      double sum = 0;
      for(int i = 0; i < ni; i++) {
        double v = m.get(i, j);
        sum += v * v;
      }
      sum = Math.sqrt(sum);
      for(int i = 0; i < ni; i++)
        m.set(i, j, m.get(i, j) / sum);
    }
  }

  ////////////////////////////////////////////////////////////
  // Methods that depend on a kernel

  public static <T> Matrix computeKernelMatrix(Kernel<T> k, T[] data) {
    int N = data.length;
    Matrix K = new Matrix(N, N);
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
        K.set(i, j, k.dot(data[i], data[j]));
    return K;
  }

  // Return the L2-norm of a vector represented by alpha: O(n^2) time
  public static <T> double norm(double[] alpha, T[] data, Kernel<T> k) {
    double sum = 0;
    for(int i = 0; i < data.length; i++)
      for(int j = 0; j < data.length; j++)
        sum += alpha[i]*alpha[j] * k.dot(data[i], data[j]);
    return Math.sqrt(sum);
  }

  public static Matrix diagonal(double[] m) {
    Matrix mat = new Matrix(m.length, m.length);
    for (int i=0; i < m.length; ++i) { mat.set(i,i, m[i]); }
    return mat;
  }
}
