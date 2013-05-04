package kernelcca;

import java.util.ArrayList;
import java.util.List;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import fig.basic.ListUtils;

public class KernelPCA<T> {

	public enum Method { PRIMAL, CHOLESKY }

	private List<T> data ;
	private Kernel<T> kernel ;
	private double eta = 0.1 ;
	private Method method = Method.CHOLESKY;
	
	public KernelPCA() {

	}

	public void setData(List<T> data, Kernel<T> kernel) {
		this.data = data;
		this.kernel = kernel;
	}
	
	private RepresentingKernel<T> solveCholesky() {
		T[] X = (T[]) new Object[data.size()];
		for (int i=0; i<data.size();i++) X[i] = data.get(i);
		IncompleteCholeskyDecomposition<T> decomp = new IncompleteCholeskyDecomposition<T>(kernel,X,eta);
		List<double[]> repnData = new ArrayList<double[]>();
		for (int i=0; i < data.size(); ++i) {
			double[] repn = decomp.getRepn(i);
			repnData.add(repn);
		}
		KernelPCA<double[]> kernelPCA = new KernelPCA<double[]>();
		kernelPCA.method = Method.PRIMAL;
		kernelPCA.setData(repnData, new LinearKernel());
		PrimalOutputKernel primalOutputKernel = (PrimalOutputKernel) kernelPCA.solve();
		return (RepresentingKernel<T>) new CholeskyOutputKernel(primalOutputKernel, decomp);
	}
	
	private class PrimalOutputKernel implements RepresentingKernel<T> {

		double[][] A ;
		
		private PrimalOutputKernel(double[][] A) {
			this.A = A;
		}
		
		public double dot(T x, T y) {
			double sum = 0.0;
			for (double[] row: A) {
				sum += ListUtils.dot(row, (double[]) x) * ListUtils.dot(row, (double[]) y); 
			}
			return sum ;
		}

		public int getD() {
			return A.length;
		}

		public double[] getRepn(int i) {
			throw new UnsupportedOperationException();
		}

		public double[] getRepn(T x) {
			double[] res = new double[getD()];
			for (int i =0; i < getD(); ++i) {
//				res[i] = ListUtils.dot(A[i], (double[]) x);
				res[i] = ListUtils.dot(ListUtils.subArray(A[i], 0, ((double[]) x).length), (double[]) x);
			}
			return res;
		}
		
	}
	
	private class CholeskyOutputKernel implements RepresentingKernel<T> {

		private PrimalOutputKernel poKernel ;
		private IncompleteCholeskyDecomposition<T> decomp ;
		
		private CholeskyOutputKernel(PrimalOutputKernel poKernel, IncompleteCholeskyDecomposition<T> decomp) {
			this.poKernel = poKernel;
			this.decomp = decomp;
		}
		
		public double dot(T x, T y) {
			double[] xVec = decomp.getRepn(x);
			double[] yVec = decomp.getRepn(y);
			return poKernel.dot((T)xVec,(T)yVec);
		}

		public int getD() {
			return decomp.getD();
		}

		public double[] getRepn(int i) {
			throw new UnsupportedOperationException();
		}

		public double[] getRepn(T x) {
			double[] vec = decomp.getRepn(x);
			return poKernel.getRepn((T)vec);
		}
		
	}
	
	private RepresentingKernel<T> solvePrimal() {
		assert kernel instanceof LinearKernel;
		int N = data.size();
		Matrix C = new Matrix(N,N);
		for (int i=0; i < N; ++i) {
			for (int j=0; j < N; ++j) {
				double[] Xi = (double[]) data.get(i);
				double[] Xj = (double[]) data.get(j);
				double Cij = 1.0/N * kernel.dot((T)Xi,(T)Xj);
				C.set(i,j, Cij);
			}
		}		
		EigenvalueDecomposition eig = new EigenvalueDecomposition(C);
		double[] eigenValues = eig.getRealEigenvalues();
		double[] absEigenValues = new double[eigenValues.length];
		for (int i=0; i < eigenValues.length; ++i) { absEigenValues[i] = Math.abs(eigenValues[i]); }
		int[] perm = ListUtils.sortedIndices(absEigenValues, true);
		Matrix V = eig.getV(); // Columns are eigenvectors
		double[][] A = new double[perm.length][];
		for (int i=0; i < A.length; ++i) {
			int j = perm[i];
			double[] column = MatrixUtils.getColumn(V, j);
			A[i] = column;
		}
		return new PrimalOutputKernel(A);
	}

	public RepresentingKernel<T> solve() {
		if (kernel instanceof LinearKernel) {
			method = Method.PRIMAL;			
		}
		
		switch (method) {
			case CHOLESKY:
				return solveCholesky();
			case PRIMAL:
				return solvePrimal();
			default:
				throw new RuntimeException();
		}
	}

	public static void main(String[] args) {
		KernelPCA<double[]> pca = new KernelPCA<double[]>();
		Matrix M = Matrix.identity(100, 100);
		Matrix A = Matrix.random(100, 100);
		Matrix X = A.times(M);
		List<double[]> data = new ArrayList<double[]>();
		for (int i=0; i < 100; ++i) {
			data.add(MatrixUtils.getRow(X,i));
		}
		pca.setData(data, new LinearKernel());
		VectorRepresenter<double[]> pcaRepresentor = pca.solve();
		for (int i=0; i < 100; ++i) {
			double[] input = data.get(i);
			double[] output = pcaRepresentor.getRepn(input);
			boolean ignore = true;
		}
	}
}
