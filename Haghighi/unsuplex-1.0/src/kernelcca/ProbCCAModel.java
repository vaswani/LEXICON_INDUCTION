package kernelcca;

import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

/**
 * Probabilistic CCA model.
 */
public class ProbCCAModel implements PairModel {
  public enum ScoreType { logProb, negL2Dist, dotProd };

  // Parameters
  public static class Options {
    @Option(gloss="Tradeoff between correlation (0) and covariance (1) for CCA")
    public double tau = 0.1; 
    @Option(gloss="Number of bases to extract for CCA")
    public int maxB = Integer.MAX_VALUE;
    @Option(gloss="Verbosity level")
    public int verbose = 0;
    @Option(gloss="Type of score")
    public ScoreType scoreType = ScoreType.logProb;
    @Option(gloss="Normalize some things to get back to the original")
    public boolean oldNormHacks;
    @Option(gloss="Do not scale by correlations")
    public boolean discardCorrelations = false;
    @Option(gloss="Normalize all projections")
    public boolean normalizeProjections = false;
    
    
  }
  public Options options;

  // Input
  int N; // Number of data points
  int Dx, Dy; // Dimensionality of X and Y
  double[][] X, Y; // (X, Y) data points
  double[] weights; // Optional weights of the data points

  Matrix Cxx, Cyy, Cxy; // Covariance matrices
  int B; // Number of bases actually extracted (<= maxB)

  // Output
  double[] lambda; // Eigenvalues for the bases (principal correlations)
  // Note: u and w are actually transposed here
  double[][] ux, uy; // basis id -> vector (principal directions)
  Matrix margSigmaInv; // Covariance for (x, y) marginalizing out z

  // Map x to E(z | x) = M U^T x
  // Cache for projecting to latent space
	Map<double[],double[]> cache1 = new HashMap<double[],double[]>();
	Map<double[],double[]> cache2 = new HashMap<double[],double[]>();

  public void setWeights(double[] weights) {
    for(int i = 0; i < N; i++)
      assert weights[i] >= 0;
    assert weights.length == N;
    this.weights = weights;
  }

  // Normalize L2 norm
  private static void vecNormalize(double[] vec) {
    double sumSq = 0.0;
    for (double x: vec) { sumSq += x * x; }
    double len = Math.sqrt(sumSq);
    assert len > 0.0;
    ListUtils.multMut(vec, 1.0/len);      
  }

  ////////////////////////////////////////////////////////////

  public ProbCCAModel(Options options) {
    this.options = options;
  }

  public void learn(List<Pair<double[],double[]>> points) {
    track(String.format("solve(): %d points", points.size()), true);
    this.N = points.size();
    this.X = new double[N][];
    this.Y = new double[N][];
    for(int i = 0; i < N; i++) {
      X[i] = points.get(i).getFirst();
      Y[i] = points.get(i).getSecond();
      this.Dx = X[0].length;
      this.Dy = Y[0].length;
    }

    // Number of bases t oget
    this.B = Math.min(options.maxB, Math.min(Dx, Dy));
    this.lambda = new double[B];
    this.ux = new double[B][];
    this.uy = new double[B][];

    // Set up problem
    Matrix Cxx = new Matrix(Dx, Dx);
    Matrix Cyy = new Matrix(Dy, Dy);
    Matrix Cxy = new Matrix(Dx, Dy);
    double sumw = (weights != null ? ListUtils.sum(weights) : N);
    for(int i = 0; i < N; i++) {
      double[] Xi = (double[])X[i];
      double[] Yi = (double[])Y[i];
      double w = (weights != null ? weights[i] : 1);
      for(int dx = 0; dx < Dx; dx++)
        for(int dy = 0; dy < Dy; dy++)
          Cxy.set(dx, dy, Cxy.get(dx, dy) + w*Xi[dx]*Yi[dy]/sumw);
      for(int dx = 0; dx < Dx; dx++)
        for(int dx2 = 0; dx2 < Dx; dx2++)
          Cxx.set(dx, dx2, Cxx.get(dx, dx2) + w*Xi[dx]*Xi[dx2]/sumw);
      for(int dy = 0; dy < Dy; dy++)
        for(int dy2 = 0; dy2 < Dy; dy2++)
          Cyy.set(dy, dy2, Cyy.get(dy, dy2) + w*Yi[dy]*Yi[dy2]/sumw);
    }
    Matrix Cyx = Cxy.transpose();

    // Save for probabilistic version
    if(options.scoreType == ScoreType.logProb) {
      this.Cxx = Cxx;
      this.Cxy = Cxy;
      this.Cyy = Cyy;
    }
    // Bx = (1-tau)*Cxx + tau*I
    // By = (1-tau)*Cyy + tau*I
    Matrix Bx = Cxx.times(1-options.tau).plus(Matrix.identity(Dx,Dx).times(options.tau));
    Matrix By = Cyy.times(1-options.tau).plus(Matrix.identity(Dy,Dy).times(options.tau));

    // BC = By\Cyx
    // E = (Bx\Cxy)*By\Cyx  ~ (Cxx \ Cxy) * (Cyy\Cyx) which is similar to (3.4) in Hardoon's paper
    Matrix BC = By.inverse().times(Cyx);
    Matrix E  = Bx.inverse().times(Cxy).times(BC);

    EigenvalueDecomposition eig = E.eig();
    double[] eigenvalues = eig.getRealEigenvalues();
    int[] perm = ListUtils.sortedIndices(eigenvalues, true);
    Matrix V = eig.getV(); // Columns are eigenvectors
    for(int b = 0; b < B; b++) {
      int j = perm[b];
      lambda[b] = Math.sqrt(eigenvalues[j]+1e-10);
      Matrix col = V.getMatrix(0, Dx-1, j, j);
      ux[b] = MatrixUtils.getColumn(col, 0);
      uy[b] = MatrixUtils.getColumn(BC.times(col).times(1/lambda[b]), 0);
      if(options.verbose >= 1)
        logs(eigenvalues[j] + " | " + Fmt.D(ux[b]) + " | " + Fmt.D(uy[b]));
    }
    logs("Canonical correlations: " + Fmt.D(lambda));

		// Normalize
    // Aria - Don't Need
//    if(false) { //options.oldNormHacks) {
//      for(int b = 0; b < B; b++) {
//        MatrixUtils.metricNormalizeMut(ux[b], Bx);
//        MatrixUtils.metricNormalizeMut(uy[b], By);
//      }
//    }
	}

  public double getScore(double[] x1, double[] x2) {
    switch(options.scoreType) {
      case logProb: return getLogProb(x1, x2);
      case negL2Dist: return getProjNegL2Dist(x1, x2);
      case dotProd: return getProjDotProd(x1, x2);
      default: throw Exceptions.unknownCase;
    }
  }

  public double getLogProb(double[] x1, double[] x2) {
    // TODO: cache values that allow this computation
    // But first, let's see if this works
    assert x1.length == Dx;
    assert x2.length == Dy;

    // Compute log p(x1, x2), which is a Normal distribution
    // with mean 0 and covariance
    if(margSigmaInv == null) {
      double[] p = lambda; // Cannonical correlations (diagonals of P)
      double[] m = ListUtils.sqrt(p); // diagonals of M1 and M2
      // W_i = \Sigma_{ii} U_i M_i for i = 1, 2 (Di x B)
      Matrix Wx = Cxx.times(new Matrix(ux).transpose()).times(MatrixUtils.diagonal(m));
      Matrix Wy = Cyy.times(new Matrix(uy).transpose()).times(MatrixUtils.diagonal(m));
      
      margSigmaInv = MatrixUtils.block(
        Cxx, Wx.times(Wy.transpose()),
        Wy.times(Wx.transpose()), Cyy).inverse();
    }

    // Note: compute up to a constant (\log det(margSigmaInv))
    return -MatrixUtils.transformedNormSquared(
      ListUtils.concat(x1, x2), margSigmaInv);
  }

  // P^{1/2} U^T x
  public double[] getProj(double[][] u, double[] p, double[] x) {
    double[] z = new double[B];
    for(int i = 0; i < B; i++) {
      if(options.oldNormHacks || !options.discardCorrelations)
        z[i] = ListUtils.dot(u[i], x);
      else
        z[i] = ListUtils.dot(u[i], x) * Math.sqrt(p[i]);
    }
    if(options.oldNormHacks || options.normalizeProjections) vecNormalize(z);
    return z;
  }
  public double[] getProj1(double[] x) {
    double[] z = cache1.get(x);
    if(z == null) {
      z = getProj(ux, lambda, x);
      cache1.put(x, z);
    }
    return z;
  }
  public double[] getProj2(double[] x) {
    double[] z = cache2.get(x);
    if(z == null) {
      z = getProj(uy, lambda, x);
      cache2.put(x, z);
    }
    return z;
  }

  public double getProjNegL2Dist(double[] x1, double[] x2) {
    return -NumUtils.l2Dist(getProj1(x1), getProj2(x2));
  }
  public double getProjDotProd(double[] x1, double[] x2) {
//	  assert x1.length == x2.length;
	  return ListUtils.dot(getProj1(x1), getProj2(x2));
  }

  public static void main(String[] args) {
    ProbCCAModel.Options options = new ProbCCAModel.Options();
    Execution.init(args, options);

    ProbCCAModel model = new ProbCCAModel(options);
    List<Pair<double[],double[]>> points = new ArrayList();
    //points.add(new Pair(new double[] {2, 3}, new double[] {4,-1}));
    //points.add(new Pair(new double[] {0, 1}, new double[] {-4,1}));
    //points.add(new Pair(new double[] {2, 1}, new double[] {-3,1}));
    points.add(new Pair(new double[] {1, 10}, new double[] {50}));
    points.add(new Pair(new double[] {10, 1}, new double[] {10}));
    model.learn(points);
    for(Pair<double[],double[]> p : points)
      logs(model.getScore(p.getFirst(), p.getSecond()));
    //logs(model.getScore(new double[] {2, 3}, new double[] {4,-1}));
    //logs(model.getScore(new double[] {0, 1}, new double[] {-4,1}));
    //logs(model.getScore(new double[] {0, 0}, new double[] {0,0}));

    Execution.finish();
  }
}
