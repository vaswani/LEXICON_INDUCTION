package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

public class KernelCCATester {
  public enum KernelType { linear, gaussian };

  Options o;

  public static class Options {
    @Option public int testCase = 1;
    @Option public int genDz = 1;
    @Option public int genDxy = 2;
    @Option public int genN = 5;
    @Option public double genEpsilon = 0.1;
    @Option public Random random = new Random(1);
    @Option public KernelType kernelType = KernelType.linear;
    @Option public int verbose = 0;
  }

  public List<Pair<double[],double[]>> generatePoints() {
    List<Pair<double[],double[]>> points = new ArrayList();
    switch(o.testCase) {
    case 0:
      // Correlation = 1, B=1
      // Singular: primal,tau=0
      points.add(new Pair(new double[]{+1, +0}, new double[]{+0,+1}));
      //points.add(new Pair(new double[]{+4, +0}, new double[]{+0,+25}));
      break;
    case 1:
      // Correlation = 0.976, B=1
      // Singular: primal/tau=0, dual
      points.add(new Pair(new double[]{+1, +1}, new double[]{+1,+2}));
      points.add(new Pair(new double[]{+2, +2}, new double[]{+4,+8}));
      break;
    case 2:
      // Correlation = 1, B=1
      // X lies in 1D subspace, Y lies in 2D subspace (just off by a little bit)
      // When tau close to 1, wy is basically in the direction of (1, 2) to capture variance
      // When tau close to 0, wy is becomes more like (4, 0)
      // This is because the optimal solution is not unique;
      // Any direction will do, but when 8 becomes a 9 in a 3rd data point,
      // we want to throw out the second coordinate of y.
      // Singular: primal/tau=0, dual
      points.add(new Pair(new double[]{+1, +1}, new double[]{+1,+2}));
      points.add(new Pair(new double[]{+2, +2}, new double[]{+2,+4}));
      points.add(new Pair(new double[]{+4, +4}, new double[]{+4,+9}));
      break;
    case 3:
      // 4 points in a box; correlation = 1, B=1
      // Singular: dual
      points.add(new Pair(new double[]{+2, +1}, new double[]{+1,+2}));
      points.add(new Pair(new double[]{+2, -1}, new double[]{-1,+2}));
      points.add(new Pair(new double[]{-2, +1}, new double[]{+1,-2}));
      points.add(new Pair(new double[]{-2, -1}, new double[]{-1,-2}));
      break;
    case 4:
      // For testing SCA: points arranged in V pattern
      points.add(new Pair(new double[]{+1, +2}, new double[]{+2,+1}));
      points.add(new Pair(new double[]{+2, +4}, new double[]{+4,+2}));
      points.add(new Pair(new double[]{+4, +8}, new double[]{+8,+4}));
      break;
    case 99:
      // Generate new data
      int Dx = o.genDxy, Dy = o.genDxy, Dz = o.genDz;
      int N = o.genN;
      Matrix wx = MatrixUtils.randOrthogonalMatrix(o.random, Dx, Dz);
      Matrix wy = MatrixUtils.randOrthogonalMatrix(o.random, Dy, Dz);
      MatrixUtils.normalizeColumns(wx);
      MatrixUtils.normalizeColumns(wy);
      if(o.verbose >= 1) {
        MatrixUtils.logMatrix("generate wx", wx);
        MatrixUtils.logMatrix("generate wy", wy);
      }
      for(int i = 0; i < N; i++) {
        double eps = o.genEpsilon;
        Matrix z = MatrixUtils.randGaussianMatrix(o.random, Dz, 1);
        Matrix zx = z.plus(noiseVector(Dz, /*i+1*/0));
        Matrix zy = z.plus(noiseVector(Dz, /*i+1*/0));
        double[] x = MatrixUtils.getColumn(wx.times(zx).plus(noiseVector(Dx, 1)), 0);
        double[] y = MatrixUtils.getColumn(wy.times(zy).plus(noiseVector(Dy, 1)), 0);
        points.add(new Pair(x, y));
      }
      break;
    default:
      throw Exceptions.unknownCase;
    }

    // Print out points
    if(o.verbose >= 2) {
      track(String.format("%d points", points.size()), true);
      for(Pair<double[],double[]> p : points)
        logs(Fmt.D(p.getFirst()) + " ||| " + Fmt.D(p.getSecond()));
      end_track();
    }

    return points;
  }

  private Matrix noiseVector(int n, double noiseFactor) {
    return MatrixUtils.randGaussianMatrix(o.random, n, 1).times(o.genEpsilon*noiseFactor);
  }

  private Kernel<double[]> getKernel() {
    switch(o.kernelType) {
      case linear: return new LinearKernel();
      case gaussian: return new GaussianKernel(1);
      default: throw Exceptions.unknownCase;
    }
  }

  public KernelCCATester(Options o, KernelCCA.Options kccaOptions) {
    this.o = o;
    KernelCCA kernelCCA = new KernelCCA(kccaOptions);
    List<Pair<double[],double[]>> points = generatePoints();
    Kernel<double[]> k = getKernel();
    kernelCCA.setData(points, k, k, null);
    kernelCCA.solve();
    kernelCCA.printSolution();
  }

  public static void main(String[] args) {
    KernelCCA.Options kccaOptions = new KernelCCA.Options();
    Options options = new Options();
    Execution.init(args, options, "cca", kccaOptions);
    KernelCCATester tester = new KernelCCATester(options, kccaOptions);
    Execution.finish();
  }
}
