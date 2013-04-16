package kernelcca;

import fig.basic.*;
import Jama.*;

/**
 * Purpose: create a vectorial representation of the N data points such that:
 *  - their inner products are preserved (K)
 *  - the dimensionality of the vectors is not larger than it needs to be
 */
public class IncompleteCholeskyDecomposition<T> implements VectorRepresenter<T> {
	// Input
	private Kernel<T> k;
	private T[] data;
	private int N, D;

	// Parameters
	private double eta; // Threshold for residual

    public Kernel getK() {
        return k;
    }

    public int getN() {
        return N;
    }

    public double getEta() {
        return eta;
    }

    public double[][] getR() {
        return R;
    }

    public double[][] getRT() {
        return RT;
    }

    public int[] getPerm() {
        return perm;
    }

    public double[] getNu() {
        return nu;
    }

    // Output
	private double[][] R; // j, i; each column is a representation of data point i
	private double[][] RT; // R Transpose
	private int[] perm; // dimension j -> data point index i
	private double[] nu; // dimension j -> residual



	public IncompleteCholeskyDecomposition(Kernel<T> k, T[] data, double eta) {
		this.N = data.length;
		this.k = k;
		this.data = data;
		this.eta = eta;

		this.R = new double[N][];
		this.perm = new int[N];
		this.nu = new double[N];

		double[] d = new double[N];
		for(int i = 0; i < N; i++) {
			d[i] = k.dot(data[i], data[i]);
		}

		int j = 0;
		while(j < N) {
			// Choose data point besti with maximum residual
			int besti = ListUtils.maxIndex(d);
			if(d[besti] <= eta) break;
			//dbg("j=%d: choose i=%d with residual d=%f", j, besti, d[besti]);

			R[j] = new double[N];
			perm[j] = besti;
			nu[j] = Math.sqrt(d[perm[j]]);

			// Reduce the residual of all the data points based on this new basis
			for(int i = 0; i < N; i++) {
				R[j][i] = k.dot(data[i], data[perm[j]]);
				for(int jj = 0; jj < j; jj++) {
					R[j][i] -= R[jj][i]*R[jj][perm[j]];
				}
				R[j][i] /= nu[j];
				d[i] -= R[j][i]*R[j][i];
				//dbg("R[j=%d,i=%d] = %f", j, i, R[j][i]);
			}
			j++;
		}
		this.D = j;

		// Remove extraneous rows (dimensions)
		R = ListUtils.subArray(R, 0, D);
		perm = ListUtils.subArray(perm, 0, D);
		nu = ListUtils.subArray(nu, 0, D);

		// R Transpose
		RT = new Matrix(R).transpose().getArray();    
	}

	public int getD() { return D; }

	// Get the D-dimensional vectorial representation of data point i
	public double[] getRepn(int i) {
		double[] r = new double[D];
		//dbg(D + " " + R[0][0]);
		for(int j = 0; j < D; j++) r[j] = R[j][i];
		return r;
	}

	// Get the D-dimensional vectorial representation of a new data point x
	public double[] getRepn(T x) {
		double[] r = new double[R.length];
		for(int j = 0; j < D; j++) {
			r[j] = k.dot(x, data[perm[j]]);
			for(int jj = 0; jj < j; jj++) {
				r[j] -= r[jj]*RT[perm[j]][jj];   //R[jj][perm[j]];
			}
			r[j] /= nu[j];
		}
		return r;
	}
}
