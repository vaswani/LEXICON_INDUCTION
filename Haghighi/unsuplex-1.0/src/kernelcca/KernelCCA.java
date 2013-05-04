package kernelcca;

import java.util.*;

import edu.berkeley.nlp.math.DoubleArrays;
import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

/**
 * Implementation(s) of kernel CCA.
 * CCA maximizes correlation between two projected views.
 * By setting tau = 1, we can maximize covariance between the two projected views.
 * By specifying kxy, we can maximize correlation/covariance
 * with respect to use to same projection,
 * which is shared correlation analysis (SCA)
 * (I just made that name up).
 * @author Percy Liang
 *
 * Usage:
 *
 * // Set parameters
 * KernelCCA.Options options = new KernelCCA.Options();
 * options.tau = 0.2; // Regularization parameter (higher is more)
 * options.maxB = 3; // Number of bases to extract
 * options.eta = 1e-4; // Controls incomplete Cholesky approximation (larger is faster but more approximate)
 * // ... create list of points, kernel kx for the X space, ky for the Y space ...
 * KernelCCA cca = new KernelCCA(options);
 * cca.setData(points, kx, ky, null);
 * Kernel<T> ko = cca.solve(); // Solve returns output kernel
 * ko.dot(x, y); // Measure "goodness" of (x, y) under the model
 *
 * TODO: probabilistic version
 * TODO: support weighted examples
 */
public class KernelCCA<T> {
	// Primal methods assume data points are vectors
	// Dual methods just require a kernel function between points
	public enum Method { primal, dual, cholesky };

	// Parameters
	public static class Options {
		@Option(gloss="Weight given to CCA")
		public double ccaWeight = 1.0;
		@Option(gloss="Weight given to shared correlation analysis (SCA)")
		public double scaWeight = 1.0;
		@Option(gloss="Tradeoff between correlation (0) and covariance (1) for CCA")
		public double tau = 0.8; 
		@Option(gloss="Tradeoff between correlation (0) and covariance (1) for SCA")
		public double scaTau = 1; 
		@Option(gloss="Number of bases to extract for CCA")
		public int maxB = Integer.MAX_VALUE;
		@Option(gloss="Number of bases to extract for SCA")
		public int scaMaxB = Integer.MAX_VALUE;
		@Option(gloss="Method to use to solve CCA (Cholesky works)")
		public Method method = Method.cholesky;
		@Option(gloss="Threshold for stopping the incomplete Cholesky")
		public double eta = 1e-4; 
		@Option(gloss="When using primal or dual methods, whether to solve for y using x (assymetric is faster)")
		public boolean symmetric = false;
		@Option(gloss="Whether to normalize representations for CCA")
		public boolean ccaNormalize = false;
		@Option(gloss="Whether to normalize representations for SCA")
		public boolean scaNormalize = false;
		@Option(gloss="Verbosity level")
		public int verbose = 0;
		@Option(gloss="Return a output kernel that gives p(x, y)")
		public boolean probabilistic = false;
		@Option(gloss="Variance for prior p(z)")
		public double priorVar = 1;
		@Option(gloss="Variance for likelihood p(x | z); we could learn this")
		public double likelihoodVar = 1;
		@Option(gloss="Weight bases when computing output kernel")
		public boolean weightBases = false;
	}
	public Options options;

	// Input (set by setData())
	int N; // Number of data points
	int Dx, Dy; // Dimensionality of X and Y (if primal
	T[] X, Y; // (X, Y) data points
	double[] weights; // Optional weights of the data points
	Kernel<T> kx, ky; // Kernels specific to the X and Y spaces
	Kernel<T> kxy; // Arbitrary similarity function on X and Y pairs

	// kx, ky, and kxy together specify a kernel on the union space of X and Y

	// Output (set by solve())
	int B; // Number of bases actually extracted (<= maxB)
	double[] lambda; // Eigenvalues for the bases
	double[][] wx, wy; // basis id -> vector (for linear kernels only)
  Matrix Cxx, Cyy; // Covariance matrices
	double[][] alphax, alphay; // basis id, i alpha coefficient of the ith example
	Kernel<T> ko; // Output correlation "kernel"

	// Representation of the two spaces (includes both CCA and SCA)
	VectorRepresenter<T> xRepn, yRepn;

	public void setNumberOfBases(int B) { this.B = B; }
	public int getNumberOfBases() { return B; }

	public double getBasisWeight(int b) {
    return options.weightBases ? lambda[b] : 1;
  }

	public void setWeights(double[] weights) {
		for(int i = 0; i < N; i++)
			assert weights[i] >= 0;
			assert weights.length == N;
			this.weights = weights;
	}

	/**
	 * Given an (x, y) pair, project onto the two subspaces and compute
	 * the dot product between the two projections.
	 * Primal and dual don't support SCA; use CholeskyOutputKernel instead.
	 */
	class PrimalOutputKernel implements Kernel<T> {
    public double dot(T x, T y) {
      if(options.probabilistic)
//        return probabilisticDot(x, y);
//    	  return probabilisticDotAria(x, y);
    	  return probabilisticDotAriaNew(x, y);
      else
        return dotDot(x, y);
    }

		public double dotDot(T x, T y) {
			double sum = 0;
			for(int b = 0; b < B; b++)
				sum += getBasisWeight(b) *
          ListUtils.dot(wx[b], (double[])x) * ListUtils.dot(wy[b], (double[])y);
			return sum;
		}

	private Matrix invC ;
	private Matrix W1 ;
	private Matrix W2 ;
	
	private Matrix getW1() {
		if (W1 == null) {
			double[] p = lambda; // Cannonical correlations (diagonals of P)
			double[] m = ListUtils.sqrt(p); // diagonals of M1 and M2
			W1 = Cxx.times(new Matrix(wx).transpose()).times(MatrixUtils.diagonal(m));
		}
		return W1;
	}
	
	private Matrix getW2() {
		if (W2 == null) {
			double[] p = lambda; // Cannonical correlations (diagonals of P)
			double[] m = ListUtils.sqrt(p); // diagonals of M1 and M2
			W2 = Cyy.times(new Matrix(wy).transpose()).times(MatrixUtils.diagonal(m));
		}
		return W2;
	}
	
	
	
	private Matrix getInverseC() {
		if (invC == null) {
			Matrix W1 = getW1();
			Matrix W2 = getW2();
									
			Matrix W = new Matrix(Dx + Dy,B);			
			W.setMatrix(0, Dx-1, 0, B-1, W1);			
			W.setMatrix(Dx, Dx+Dy-1, 0, B-1, W2);
			
			double sigma = options.likelihoodVar;
			Matrix sigmaSquaredI = Matrix.identity(B, B).times(sigma*sigma);
			Matrix M = W.transpose().times(W).plus(sigmaSquaredI);			
			
			Matrix invM = M.inverse();
			Matrix invSigmaI = Matrix.identity(Dx+Dy, Dx+Dy).times(1.0/sigma);
			Matrix outer = W.times(invM).times(W.transpose()).times(1.0/(sigma*sigma));
			Matrix C = invSigmaI.minus(outer);
			invC = C.inverse();;
		}
		return invC;
	}
	
	private Matrix columnStack(double[] x, double[] y) {
		Matrix m = new Matrix(1,Dx+Dy);
		for (int i=0; i < Dx; ++i) {
			m.set(0, i, x[i]);
		}
		for (int i=0; i < Dy; ++i) {
			m.set(0, i+Dx, y[i]);
		}
		return m;
	}
	                                                
	
		
	public double probabilisticDotAria(T _x, T _y) {
		double[] xVec = (double[]) _x;
		double[] yVec = (double[]) _y;
	    Matrix invC = getInverseC();
	    Matrix x = columnStack(xVec, yVec);
	    // outer product x C^{-1} x'
	    Matrix res = x.times(invC).times(x.transpose());
	    assert res.getRowDimension() == 1 && res.getColumnDimension() == 1;
	    return -res.get(0,0);		
	}
	
	public double probabilisticDotAriaNew(T _x, T _y) {
	      double[] x = (double[])_x;
	      double[] y = (double[])_y;

	      // Project x and y onto bases: xz = wx^T x
	      double[] xz = new double[B];
	      double[] yz = new double[B];
	      for(int b = 0; b < B; b++) {
			xz[b] = ListUtils.dot(wx[b], x);
			yz[b] = ListUtils.dot(wy[b], y);
	      }

	      // Some convenient variables
	      double[] p = lambda; // Cannonical correlations (diagonals of P)
	      double[] m = ListUtils.sqrt(p); // diagonals of M1 and M2
	      double[] pp = new double[B]; // (I - P^2)^{-1}
	      for(int b = 0; b < B; b++) pp[b] = 1/(1-p[b]*p[b]);

	      // Compute:
	      //   mz = E(z | x, y), which combines the guesses of xz and yz
	      // Equation on page 4 of Bach/Jordan's CCA paper:
	      // Note that their U_1d is our wx and their U_2d is our wy
	      // Intuition: weigh the components of z based on their correlation
	      double[] mz = new double[B];
	      for(int b = 0; b < B; b++) {
	        // These updates from the Bach/Jordan paper don't make sense because
	        // when correlation is 0, somehow the mean is infinity
	        mz[b] = m[b] * pp[b]*(1+p[b]) * (xz[b]+yz[b]);
	      }
		Matrix res1 = getW1().times(new Matrix(mz,mz.length)).transpose();
		assert res1.getRowDimension() == 1;
		double[] predX = res1.getArray()[0];
		double[] predY = getW2().times(new Matrix(mz,mz.length)).transpose().getArray()[0];
		double joint = -NumUtils.l2Dist(x, predX) -NumUtils.l2Dist(y, predY);
		return joint;
	}
	
	
		
    public double probabilisticDot(T _x, T _y) {
      double[] x = (double[])_x;
      double[] y = (double[])_y;

      // Project x and y onto bases: xz = wx^T x
      double[] xz = new double[B];
      double[] yz = new double[B];
			for(int b = 0; b < B; b++) {
				xz[b] = ListUtils.dot(wx[b], x);
        yz[b] = ListUtils.dot(wy[b], y);
      }

      // Some convenient variables
      double[] p = lambda; // Cannonical correlations (diagonals of P)
      double[] m = ListUtils.sqrt(p); // diagonals of M1 and M2
      double[] pp = new double[B]; // (I - P^2)^{-1}
			for(int b = 0; b < B; b++) pp[b] = 1/(1-p[b]*p[b]);

//      dbgs(Fmt.D(p));
      for(int b = 0; b < B; b++) {
        assert p[b] >= 0 && p[b] <= 1 : p[b];
        assert m[b] >= 0 && m[b] <= 1 : m[b];
      }

      // Compute:
      //   mz = E(z | x, y), which combines the guesses of xz and yz
      //   vz = var(z | x, y), which is how uncertain we are about mz
      // Equation on page 4 of Bach/Jordan's CCA paper:
      // Note that their U_1d is our wx and their U_2d is our wy
      // Intuition: weigh the components of z based on their correlation
      double[] mz = new double[B];
      double[] vz = new double[B];
			for(int b = 0; b < B; b++) {
        // These updates from the Bach/Jordan paper don't make sense because
        // when correlation is 0, somehow the mean is infinity
        mz[b] = m[b] * pp[b]*(1+p[b]) * (xz[b]+yz[b]);
        vz[b] = m[b] * pp[b]*(1+p[b]) * (m[b]+m[b]);
        // Make up my own hack:
//        mz[b] = (xz[b] + yz[b])/2; // Average
//        vz[b] = 1;
      }
      if(options.verbose >= 2) {
        dbgs("xz = %s", Fmt.D(xz));
        dbgs("yz = %s", Fmt.D(yz));
        dbgs("mz = %s", Fmt.D(mz));
        dbgs("vz = %s", Fmt.D(vz));
      }

      // mx and my are the mean vectors for x, y given z
      // Compute mx = Cxx wx M z, my = Cyy wy M z
      double[] mx = new double[Dx];
      double[] my = new double[Dy];
      for(int b = 0; b < B; b++) ListUtils.incr(mx, m[b]*mz[b], wx[b]);
      for(int b = 0; b < B; b++) ListUtils.incr(my, m[b]*mz[b], wy[b]);
      mx = MatrixUtils.getColumn(Cxx.times(new Matrix(mx, Dx)), 0);
      my = MatrixUtils.getColumn(Cyy.times(new Matrix(my, Dy)), 0);

      // Evaluate log p(x, y, z) (up to a constant)
      double joint = 0;
      joint -= NumUtils.l2Norm(mz) / options.priorVar; // p(z)
      joint -= NumUtils.l2Dist(mx, x) / options.likelihoodVar; // p(x | z)
      joint -= NumUtils.l2Dist(my, y) / options.likelihoodVar; // p(y | z)
      if(options.verbose >= 2) dbgs("joint = %f", joint);

      // Evaluate log p(z | x, y) (up to a constant)
      // At mean, no contribution from exp term
      // Strange: posterior variance is diagonal,
      // so log det is just sum of diagonal entries
      double posterior = -ListUtils.sum(vz);
      if(options.verbose >= 2) dbgs("posterior = %f", posterior);

      // Note: we return log p(x, y) up to a constant
      // (which does depend on the parameters)      
      return joint - posterior;
    }
	}

	class DualOutputKernel implements Kernel<T> {
		public double dot(T x, T y) {
			double sum = 0;
			for(int b = 0; b < B; b++) {
				double lx = 0;
				for(int i = 0; i < N; i++)
					lx += alphax[b][i]*kx.dot(X[i], x);
				double ly = 0;
				for(int i = 0; i < N; i++)
					ly += alphay[b][i]*ky.dot(Y[i], y);
				sum += lx*ly;
			}
			return sum;
		}
	}

	private static void vecNormalize(double[] vec) {
		double sumSq = 0.0;
		for (double x: vec) { sumSq += x * x; }
		double len = Math.sqrt(sumSq);
		assert len > 0.0;
		DoubleArrays.scale(vec, 1.0/len);			
	}

	class CholeskyOutputKernel implements Kernel<T> {
		VectorRepresenter<T> decompx, decompy; // for CCA
		VectorRepresenter<TypedPoint<T>> decomp; // For SCA

		Kernel<double[]> ko; // Output kernel of CCA
		double[][] ws; // SCA: basis index b -> vector

		public CholeskyOutputKernel(VectorRepresenter<T> decompx,
				VectorRepresenter<T> decompy, Kernel<double[]> ko) {
      // Actually, we don't want to cache these these representations,
      // but rather these representations dotted with wx and wy
			this.decompx = new CachingVectorRepresenter<T>(decompx);
			this.decompy = new CachingVectorRepresenter<T>(decompy);
			this.ko = ko;
		}

		// Can be slow: use CholeskyRepresentor if possible
		public double dot(T x, T y) {
			// CCA contribution
			double ccaSum = 0;
			if(ko != null) {
				double[] px = decompx.getRepn(x);
				double[] py = decompy.getRepn(y);
				if(options.ccaNormalize) vecNormalize(px);
				if(options.ccaNormalize) vecNormalize(py);
				ccaSum = ko.dot(px, py);
			}

			// SCA contribution
			double scaSum = 0;
			if(ws != null) {
				double[] px = decomp.getRepn(TypedPoint.newX(x));
				double[] py = decomp.getRepn(TypedPoint.newY(y));
				if(options.scaNormalize) vecNormalize(px);
				if(options.scaNormalize) vecNormalize(py);
				for(int b = 0; b < ws.length; b++)
					scaSum += ListUtils.dot(ws[b], px) * ListUtils.dot(ws[b], py);
			}

			return options.ccaWeight * ccaSum + options.scaWeight * scaSum;
		}
	}

	/**
	 * Represents either X or Y points as a linear vector which
	 * incorporates both CCA and SCA components.
	 * The representation should be such that when we take the dot product
	 * of the representations, we get the same answer as if we call dot()
	 * on the CholeskyOutputKernel.
	 * Using linear representations will just be a lot faster
	 * because we have to call dot a lot of times.
	 * Variable names ending in x could apply to y
	 */
	class CholeskyRepresentor implements VectorRepresenter<T> {
		VectorRepresenter<T> decompx;
		VectorRepresenter<TypedPoint<T>> decomp;
		TypedPoint.PointType pointType;
		double[][] wx; // CCA
		double[][] ws; // SCA

		public CholeskyRepresentor(VectorRepresenter<T> decompx, double[][] wx, CholeskyOutputKernel ko, TypedPoint.PointType pointType) {
			this.decompx = decompx;
			this.wx = wx;
			this.decomp = ko.decomp;
			this.ws = ko.ws;
			this.pointType = pointType;
		}
		public int getD() { return wx.length + (ws != null ? ws.length : 0); }
		public double[] getRepn(int i) { throw new UnsupportedOperationException(); }

		public double[] getRepn(T xobj) {
			double[] repn = new double[getD()];

			// CCA contribution
			if(decompx != null) {
				double[] x = decompx.getRepn(xobj);
				if(options.ccaNormalize) vecNormalize(x);
				for(int b = 0; b < wx.length; b++)
					repn[b] = options.ccaWeight * ListUtils.dot(wx[b], x);
			}

			// SCA contribution
			if(decomp != null) {
				double[] x = decomp.getRepn(new TypedPoint(pointType, xobj));
				if(options.scaNormalize) vecNormalize(x);
				for(int b = 0; b < ws.length; b++)
					repn[b+wx.length] = options.scaWeight * ListUtils.dot(ws[b], x);
			}
			//dbgs(xobj + " => " + Fmt.D(repn));

			return repn;
		}
	}

	////////////////////////////////////////////////////////////

	public KernelCCA(Options options) {
		this.options = options;
	}

	private boolean isLinearKernel() {
		return kx instanceof LinearKernel && ky instanceof LinearKernel;
	}
	boolean isPrimal() { return options.method == Method.primal; }
	boolean isDual() { return options.method == Method.dual; }
	boolean isCholesky() { return options.method == Method.cholesky; }
	public boolean useSCA() { return kxy != null; }

	public void setData(List<Pair<T,T>> points, Kernel<T> kx, Kernel<T> ky, Kernel<T> kxy) {
		track(String.format("setData(): %d points", points.size()), true);
		for(Pair<T,T> p : points) {
			if(options.verbose > 0 && p.getFirst() instanceof String) {
				logs(p.getFirst() + " | " + p.getSecond());
			}
		}
		end_track();
		this.N = points.size();
		this.kx = kx; this.ky = ky; this.kxy = kxy;
		this.X = (T[])new Object[N];
		this.Y = (T[])new Object[N];
		for(int i = 0; i < N; i++) {
			X[i] = points.get(i).getFirst();
			Y[i] = points.get(i).getSecond();
			if(isLinearKernel()) {
				this.Dx = ((double[])X[0]).length;
				this.Dy = ((double[])Y[0]).length;
			}
		}
	}
	
	// Assume a linear kernel
	public void solvePrimal() {
		this.B = Math.min(options.maxB, Math.min(Dx, Dy));
		this.lambda = new double[B];
		this.wx = new double[B][Dx];
		this.wy = new double[B][Dy];

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
    this.Cxx = Cxx;
    this.Cyy = Cyy;

		Matrix Bx = Cxx.times(1-options.tau).plus(Matrix.identity(Dx,Dx).times(options.tau));
		Matrix By = Cyy.times(1-options.tau).plus(Matrix.identity(Dy,Dy).times(options.tau));
		//MatrixUtils.logMatrix("Bx", Bx);
		//MatrixUtils.logMatrix("By", By);

		if(options.symmetric) {
			Matrix I = MatrixUtils.block(Bx.inverse(), null, null, By.inverse());
			Matrix C = MatrixUtils.block(null, Cxy, Cyx, null);
			Matrix E = I.times(C);

			EigenvalueDecomposition eig = E.eig();
			double[] eigenvalues = eig.getRealEigenvalues();
			int[] perm = ListUtils.sortedIndices(eigenvalues, true);
			Matrix V = eig.getV(); // Columns are eigenvectors
			for(int b = 0; b < B; b++) {
				int j = perm[b];
				lambda[b] = eigenvalues[j];
				wx[b] = MatrixUtils.getColumn(V, j, 0, Dx);
				wy[b] = MatrixUtils.getColumn(V, j, Dx, Dx+Dy);
			}
		}
		else {
			Matrix BC = By.inverse().times(Cyx);
			Matrix E = Bx.inverse().times(Cxy).times(BC);

			EigenvalueDecomposition eig = E.eig();
			double[] eigenvalues = eig.getRealEigenvalues();
			int[] perm = ListUtils.sortedIndices(eigenvalues, true);
			Matrix V = eig.getV(); // Columns are eigenvectors
			for(int b = 0; b < B; b++) {
				int j = perm[b];
				lambda[b] = Math.sqrt(eigenvalues[j]);
				Matrix col = V.getMatrix(0, Dx-1, j, j);
				wx[b] = MatrixUtils.getColumn(col, 0);
				wy[b] = MatrixUtils.getColumn(BC.times(col).times(1/lambda[b]), 0);
			}
		}

		// Normalize
		for(int b = 0; b < B; b++) {
			MatrixUtils.metricNormalizeMut(wx[b], Bx);
			MatrixUtils.metricNormalizeMut(wy[b], By);
		}

		this.ko = new PrimalOutputKernel();
	}

	// B = number of bases
	// Doesn't work with singular kernel matrices.
	public void solveDual() {
		this.B = Math.min(options.maxB, N);
		this.lambda = new double[B];
		this.alphax = new double[B][N];
		this.alphay = new double[B][N];

		Matrix Kx = MatrixUtils.computeKernelMatrix(kx, X);
		Matrix Ky = MatrixUtils.computeKernelMatrix(ky, Y);
		Matrix Kxy = Kx.times(Ky);
		Matrix Kyx = Ky.times(Kx);
		Matrix Bx = Kx.times(Kx).times(1-options.tau).plus(Kx.times(options.tau));
		Matrix By = Ky.times(Ky).times(1-options.tau).plus(Ky.times(options.tau));

		if(options.symmetric) {
			Matrix I = MatrixUtils.block(Bx.inverse(), null, null, By.inverse());
			Matrix K = MatrixUtils.block(null, Kxy, Kyx, null);
			Matrix E = I.times(K);

			EigenvalueDecomposition eig = E.eig();
			double[] eigenvalues = eig.getRealEigenvalues();
			int[] perm = ListUtils.sortedIndices(eigenvalues, true);
			Matrix V = eig.getV(); // Columns are eigenvectors
			for(int b = 0; b < B; b++) {
				int j = perm[b];
				lambda[b] = eigenvalues[j];
				alphax[b] = MatrixUtils.getColumn(V, j, 0, N);
				alphay[b] = MatrixUtils.getColumn(V, j, N, N+N);
			}
		}
		else {
			Matrix BC = By.inverse().times(Kyx);
			Matrix E = Bx.inverse().times(Kxy).times(BC);

			EigenvalueDecomposition eig = E.eig();
			double[] eigenvalues = eig.getRealEigenvalues();
			int[] perm = ListUtils.sortedIndices(eigenvalues, true);
			Matrix V = eig.getV(); // Columns are eigenvectors
			for(int b = 0; b < B; b++) {
				int j = perm[b];
				lambda[b] = Math.sqrt(eigenvalues[j]);
				Matrix col = V.getMatrix(0, N-1, j, j);
				alphax[b] = MatrixUtils.getColumn(col, 0);
				alphay[b] = MatrixUtils.getColumn(BC.times(col).times(1/lambda[b]), 0);
			}
		}

		// Normalize
		for(int b = 0; b < B; b++) {
			MatrixUtils.metricNormalizeMut(alphax[b], Bx);
			MatrixUtils.metricNormalizeMut(alphay[b], By);
		}

		this.ko = new DualOutputKernel();
	}

	public void solveCholesky() {
		// Perform incomplete Cholesky decomposition and invoke a primal algorithm
		VectorRepresenter<T> decompx = new IncompleteCholeskyDecomposition<T>(kx, X, options.eta);
		VectorRepresenter<T> decompy = new IncompleteCholeskyDecomposition<T>(ky, Y, options.eta);

		List<Pair<double[],double[]>> points = new ArrayList();
		track(String.format("New representation: Dx = %d, Dy = %d", decompx.getD(), decompy.getD()), true);
		if(options.verbose >= 2) {
			for(int i = 0; i < N; i++)
				logs(Fmt.D(decompx.getRepn(i)) + " ||| " + Fmt.D(decompy.getRepn(i)));
		}
		end_track();
		for(int i = 0; i < N; i++)
			points.add(new Pair(decompx.getRepn(i), decompy.getRepn(i)));

		// Create a primal problem and solve it
		Method saveMethod = options.method; options.method = Method.primal;

		KernelCCA<double[]> cca = new KernelCCA(options);
		cca.setData(points, new LinearKernel(), new LinearKernel(), null);
		this.ko = new CholeskyOutputKernel(decompx, decompy, cca.solve());

		options.method = saveMethod;

		computeSCA();
		// These variables are just for printing out the solution
		this.B = cca.B;
		this.lambda = cca.lambda;
		this.wx = cca.wx;
		this.wy = cca.wy;

		this.xRepn = new CholeskyRepresentor(decompx, wx, (CholeskyOutputKernel)ko, TypedPoint.PointType.X);
		this.yRepn = new CholeskyRepresentor(decompy, wy, (CholeskyOutputKernel)ko, TypedPoint.PointType.Y);
	}

	public VectorRepresenter<T> getXRepresentation() {
		if (options.method != Method.cholesky) {
			throw new UnsupportedOperationException("Only allowed for Cholesky");
		}
		return xRepn;
	}

	public VectorRepresenter<T> getYRepresentation() {
		if (options.method != Method.cholesky) {
			throw new UnsupportedOperationException("Only allowed for Cholesky");
		}
		return yRepn;
	}

	public void computeSCA() {
		if(!useSCA()) return;

		track("computeSCA()");

		// Find a common representation for both X and Y
		// that preserves all the inner products
		Kernel<TypedPoint<T>> k = new TypedKernel<T>(kx, ky, kxy);
		TypedPoint[] XY = new TypedPoint[N*2];
		for(int i = 0; i < N; i++) XY[i] = TypedPoint.newX(X[i]);
		for(int i = 0; i < N; i++) XY[N+i] = TypedPoint.newY(Y[i]);
		VectorRepresenter<TypedPoint<T>> decomp =
			new IncompleteCholeskyDecomposition<TypedPoint<T>>(k, XY, options.eta);
		VectorRepresenter<TypedPoint<T>> decompx = decomp;
		VectorRepresenter<TypedPoint<T>> decompy =
			new ShiftVectorRepresenter<TypedPoint<T>>(decomp, N);

		// Optimize the SCA objective
		int D = decomp.getD();
		logs("Representing X and Y (2N = %d points) with D=%d dimensions", 2*N, D);
		Matrix Cxy = new Matrix(D, D);
		Matrix A = new Matrix(D, D);
		for(int i = 0; i < N; i++) {
			double[] x = decompx.getRepn(i);
			double[] y = decompy.getRepn(i);
			//dbgs(i + ": " + Fmt.D(x) + " ||| " + Fmt.D(y));
			for(int dx = 0; dx < D; dx++) {
				for(int dy = 0; dy < D; dy++) {
					Cxy.set(dx, dy, Cxy.get(dx, dy) + x[dx]*y[dy]/N);
					A.set(dx, dy, A.get(dx, dy) +
							(1-options.scaTau)*0.5*(x[dx]*x[dy]+y[dx]*y[dy])/N +
							options.scaTau*(dx == dy ? 1.0/N : 0));
				}
			}
		}
		EigenvalueDecomposition eig = A.inverse().times(Cxy).eig();
		double[] eigenvalues = eig.getRealEigenvalues();
		double[] imagEigenvalues = eig.getImagEigenvalues();
		int numNeg = 0;
		int numComplex = 0;
		for(int d = 0; d < D; d++) {
			if(!NumUtils.equals(imagEigenvalues[d], 0)) {
				eigenvalues[d] = Double.NEGATIVE_INFINITY;
				numComplex++;
			}
			else if(eigenvalues[d] < 0)
				numNeg++;
		}
		int[] perm = ListUtils.sortedIndices(eigenvalues, true);
		Matrix V = eig.getV(); // Columns are eigenvectors
		int B = Math.min(options.scaMaxB, D-numNeg-numComplex);
		logs("D=%d eigenvalues: %d complex, %d negative, using B=%d good bases",
				D, numComplex, numNeg, B);
		double[][] ws = new double[B][];
		for(int b = 0; b < B; b++) {
			int j = perm[b];
			ws[b] = MatrixUtils.getColumn(V, j);
			//dbgs(eigenvalues[j] + " " + Fmt.D(ws[b]));
		}

		((CholeskyOutputKernel)ko).ws = ws;
		((CholeskyOutputKernel)ko).decomp = decomp;

		end_track();
	}

	public void printSolution() {
		track("Solution", true);
		for(int b = 0; b < B; b++) {
			track(String.format("Basis %d: eigenvalue = %f", b, lambda[b]), true);
			if(options.verbose >= 2) {
				if(isPrimal()) {
					// Compute alphas as dot products with data points
					// (this is possible by the representer theorem)
					double[] alphax = new double[N];
					double[] alphay = new double[N];
					for(int i = 0; i < N; i++) {
						alphax[i] = ListUtils.dot(wx[b], (double[])X[i]) / NumUtils.l2NormSquared((double[])X[i]);
						alphay[i] = ListUtils.dot(wy[b], (double[])Y[i]) / NumUtils.l2NormSquared((double[])Y[i]);
					}
					logs("alphax = %s", Fmt.D(alphax));
					logs("alphay = %s", Fmt.D(alphay));
					logs("wx = %s", Fmt.D(wx[b]));
					logs("wy = %s", Fmt.D(wy[b]));
				}
				else if(isDual()) {
					logs("alphax = %s", Fmt.D(alphax[b]));
					logs("alphay = %s", Fmt.D(alphay[b]));
					// If linear kernel, can represent the bases in their primal space as well
					if(isLinearKernel()) {
						double[] wx = new double[Dx];
						double[] wy = new double[Dy];
						for(int i = 0; i < N; i++) ListUtils.incr(wx, alphax[b][i], (double[])X[i]);
						for(int i = 0; i < N; i++) ListUtils.incr(wy, alphay[b][i], (double[])Y[i]);
						logs("wx = %s", Fmt.D(wx));
						logs("wy = %s", Fmt.D(wy));
					}
				}
				else if(isCholesky()) {
					logs("wx = %s", Fmt.D(wx[b]));
					logs("wy = %s", Fmt.D(wy[b]));
				}
				else
					throw Exceptions.unknownCase;
			}

			end_track();
		}

    // Average doesn't make sense if probabilistic
		double sum = 0;
		track("Evaluate points", true);
		for(int i = 0; i < N; i++) {
			double v = ko.dot(X[i], Y[i]);
			if(options.verbose >= 2)
				logs(
						Fmt.D((double[])X[i]) + " ||| " + Fmt.D((double[])Y[i]) + " ==> " + v);
			sum += v;
		}
		logss("Average = " + sum/N);
		end_track();

		end_track();
	}

	public Kernel<T> solve() {
		track("solve(): " + options.method, true);

		switch(options.method) {
			case primal: solvePrimal(); break;
			case dual: solveDual(); break;
			case cholesky: solveCholesky(); break;
			default: throw Exceptions.unknownCase;
		}

		end_track();
		return ko;
	}
}
