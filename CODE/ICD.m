classdef ICD
%
%  Purpose: create a vectorial representation of the N data points such that:
%   - their inner products are preserved (K)
%   - the dimensionality of the vectors is not larger than it needs to be
%   
    methods(Static)	
        
        function model = ichol_data(data, eta)
            K = data*data';
            model = ICD.ichol(K, eta);
            model.data = data;
        end

        function model = ichol(K, eta)
            d = diag(K);
            N = size(K,1);
            j = 1;
            while (j<=N)
                [~, besti] = max(d);
                if d(besti) <= eta
                    break;
                end

                R(j,:) = zeros(1,N);
                perm(j) = besti;
                nu(j) = sqrt(d(perm(j)));

                for i = 1:N,
                    R(j,i) = K(i, perm(j));
                    for jj=1:(j-1),
     					R(j,i) = R(j,i) - R(jj,i)*R(jj,perm(j));
                    end
                    R(j,i) = R(j,i) / nu(j);
                    d(i) = d(i) - R(j,i)*R(j,i);
                end
                j = j + 1;
            end
            D = j-1;

            % Remove extraneous rows (dimensions)
            R = R(1:D,:);
            perm = perm(1:D);%
            nu = nu(1:D);

            model.K = K;
            model.R = R;
            model.RT = R';
            model.nu = nu;
            model.perm = perm;
            model.D = D;
        end
        
        function r=getRepresentations_data(data, model)
            K_X = data*model.data';
            r = ICD.getRepresentations(K_X, model);
        end

        % Get the D-dimensional vectorial representation of a new data point x
        function r = getRepresentation(K_x, model)
            D = model.D;
            r = zeros(D,1);
            for j=1:D,
                perm_j = model.perm(j);
                r(j) = K_x(perm_j);
                for jj=1:(j-1),
                    r(j) = r(j) - r(jj)*model.RT(perm_j,jj);   
                end
                r(j) = r(j) / model.nu(j);
            end
        end
        
        function r = getRepresentations(K_X, model)
            D = model.D;
            N = size(K_X,1);
            r = zeros(N,D);
            for j=1:D,
                perm_j = model.perm(j);
                r(:,j) = K_X(:,perm_j);
                for jj=1:(j-1),
                    r(:,j) = r(:,j) - r(:,jj).*model.RT(perm_j,jj);   
                end
                r(:,j) = r(:,j) ./ model.nu(j);
            end
        end
        
        
        function test()
            data = [1,3,1;1,4,1;1,-3,-5;2,2.5,2];
            K = data*data';
            eta = 0.01;
            model = ICD.ichol(K, eta);
            z1 = [3,-2,1];
            z2 = [2,1,1];
            K_x1 = z1*data';
            K_x2 = z2*data';
            K_X  = [z1;z2]*data';
            r1 = ICD.getRepresentation(K_x1,model);
            r2 = ICD.getRepresentation(K_x2,model);
            r  = ICD.getRepresentations(K_X,model);
            
            r1
            r2
            r
            eta
            nu = model.nu
            perm =model.perm
            R = model.R
            norm(R'*R-K) % should be roughly the same.
        end
    end
    
end

%         
%         package kernelcca;
% 
% import fig.basic.*;
% import Jama.*;

% /**
%  * Purpose: create a vectorial representation of the N data points such that:
%  *  - their inner products are preserved (K)
%  *  - the dimensionality of the vectors is not larger than it needs to be
%  */
% public class IncompleteCholeskyDecomposition<T> implements VectorRepresenter<T> {
% 	// Input
% 	private Kernel<T> k;
% 	private T[] data;
% 	private int N, D;
% 
% 	// Parameters
% 	private double eta; // Threshold for residual
% 
% 	// Output
% 	private double[][] R; // j, i; each column is a representation of data point i
% 	private double[][] RT; // R Transpose
% 	private int[] perm; // dimension j -> data point index i
% 	private double[] nu; // dimension j -> residual
% 
% 	public IncompleteCholeskyDecomposition(Kernel<T> k, T[] data, double eta) {
% 		this.N = data.length;
% 		this.k = k;
% 		this.data = data;
% 		this.eta = eta;
% 
% 		this.R = new double[N][];
% 		this.perm = new int[N];
% 		this.nu = new double[N];
% 
% 		double[] d = new double[N];
% 		for(int i = 0; i < N; i++) {
% 			d[i] = k.dot(data[i], data[i]);
% 		}
% 
% 		int j = 0;
% 		while(j < N) {
% 			// Choose data point besti with maximum residual
% 			int besti = ListUtils.maxIndex(d);
% 			if(d[besti] <= eta) break;
% 			//dbg("j=%d: choose i=%d with residual d=%f", j, besti, d[besti]);
% 
% 			R[j] = new double[N];
% 			perm[j] = besti;
% 			nu[j] = Math.sqrt(d[perm[j]]);
% 
% 			// Reduce the residual of all the data points based on this new basis
% 			for(int i = 0; i < N; i++) {
% 				R[j][i] = k.dot(data[i], data[perm[j]]);
% 				for(int jj = 0; jj < j; jj++) {
% 					R[j][i] -= R[jj][i]*R[jj][perm[j]];
% 				}
% 				R[j][i] /= nu[j];
% 				d[i] -= R[j][i]*R[j][i];
% 				//dbg("R[j=%d,i=%d] = %f", j, i, R[j][i]);
% 			}
% 			j++;
% 		}
% 		this.D = j;
% 
% 		// Remove extraneous rows (dimensions)
% 		R = ListUtils.subArray(R, 0, D);
% 		perm = ListUtils.subArray(perm, 0, D);
% 		nu = ListUtils.subArray(nu, 0, D);
% 
% 		// R Transpose
% 		RT = new Matrix(R).transpose().getArray();    
% 	}
% 
% 	public int getD() { return D; }
% 
% 	// Get the D-dimensional vectorial representation of data point i
% 	public double[] getRepn(int i) {
% 		double[] r = new double[D];
% 		//dbg(D + " " + R[0][0]);
% 		for(int j = 0; j < D; j++) r[j] = R[j][i];
% 		return r;
% 	}
% 
% 	// Get the D-dimensional vectorial representation of a new data point x
% 	public double[] getRepn(T x) {
% 		double[] r = new double[R.length];
% 		for(int j = 0; j < D; j++) {
% 			r[j] = k.dot(x, data[perm[j]]);
% 			for(int jj = 0; jj < j; jj++) {
% 				r[j] -= r[jj]*RT[perm[j]][jj];   //R[jj][perm[j]];
% 			}
% 			r[j] /= nu[j];
% 		}
% 		return r;
% 	}
% }


