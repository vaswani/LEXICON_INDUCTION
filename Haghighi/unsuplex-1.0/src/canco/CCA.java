package canco;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fig.basic.*;

import edu.berkeley.nlp.math.DoubleArrays;
import fig.basic.StopWatch;

public class CCA {
	
	boolean useComplexProjections = false;
	
	/**
	 * Each row is a data item
	 * @param left
	 * @param right
	 */
	public void doCCA(double[][] left, double[][] right) {
		assert left.length == right.length;
		try {
			File leftFile = File.createTempFile("leftFeatures", "matrix_txt");
			File rightFile = File.createTempFile("rightFeatures", "matrix_txt");
			
//			// temp
//			MatrixIO.writeMatrix(left, "./left.mat");
//			MatrixIO.writeMatrix(right, "./right.mat");
//			System.out.println("Kill me.");
			
			String leftMatrixPath = leftFile.getAbsolutePath();
			String rightMatrixPath = rightFile.getAbsolutePath();
			System.out.println("left: " + leftMatrixPath);
			System.out.println("right: " + rightMatrixPath);
						
			writeAsciiMatrix(left, leftMatrixPath);
			writeAsciiMatrix(right, rightMatrixPath);
			
			String uFile = File.createTempFile("uMatrix", ".matrix_txt").getAbsolutePath();
			String vFile = File.createTempFile("vMatrix", ".matrix_txt").getAbsolutePath();
			String iuFile = File.createTempFile("iuMatrix", ".matrix_txt").getAbsolutePath();
			String ivFile = File.createTempFile("ivMatrix", ".matrix_txt").getAbsolutePath();
			String rFile = File.createTempFile("rMatrix", ".matrix_txt").getAbsolutePath();
									
			String script = String.format(
 					"X = load('%s','-ASCII');\n" +
					"Y = load('%s','-ASCII');\n" +
					"X = X';\n" + 
					"Y = Y';\n" +
					//"X = X + (rand(X)*(10^-5));\n" + 
					//"Y = Y + (rand(Y)*(10^-5));\n" + 
					"z = [X;Y];\n" +
					"C = cov(z.');\n" +
					//"C = C + (rand(C)*(10^-5));\n" +
					"sx = size(X,1);\n" +
					"sy = size(Y,1);\n" + 
					"Cxx = C(1:sx, 1:sx) + 0.0000000000001*eye(sx);\n" +
					"Cxy = C(1:sx, sx+1:sx+sy);\n" + 
					"Cyx = Cxy';\n" + 
					"Cyy = C(sx+1:sx+sy, sx+1:sx+sy) + 0.0001*eye(sy);\n" +
					"invCyy = inv(Cyy);\n" + 
					"[U,r] = eig(inv(Cxx)*Cxy*invCyy*Cyx);\n" +
					"r = sqrt(real(r));\n" +
					"V = fliplr(U);\n" +
					"r = flipud(diag(r));\n"+
					"[r,I]= sort((real(r)));\n" + 
					"r = flipud(r);\n" + 
					"for j = 1:length(I)\n" + 
					  "U(:,j) = V(:,I(j));\n" +
					"end\n" + 
					"U = fliplr(U);\n" +
					"V = invCyy*Cyx*U;\n" + 
					"V = V./repmat(sqrt(sum(real(V).^2)),sy,1);\n" + 
					"rU = real(U');\n" +  
					"rV = real(V');\n" + 
					"iU = imag(U');\n" +  
					"iV = imag(V');\n" + 
					//"iU = zeros(size(U',1),size(U',2));\n" +  
					//"iV = zeros(size(V',1),size(V',2));\n" +  
					"save('-text','%s','rU');\n" +
					"save('-text','%s','rV');\n" +
					"save('-text','%s','iU');\n" +
					"save('-text','%s','iV');\n" +
					"save('-text','%s','r');\n",
					leftMatrixPath, rightMatrixPath,
					uFile,vFile,iuFile,ivFile, rFile);

			
			File matlabScript = File.createTempFile("cca_script", ".m");
			BufferedWriter br = new BufferedWriter(new FileWriter(matlabScript));
			br.write(script); br.flush(); br.close();
			System.out.println("matlab: " + matlabScript.getAbsolutePath());
			ProcessBuilder pb = new ProcessBuilder("octave", matlabScript.getAbsolutePath());
			//ProcessBuilder pb = new ProcessBuilder("matlab",  "-nosplash", "-nojvm", "-r", script);
			pb.redirectErrorStream();
			try {
				System.out.println("cmd: " + pb.command());
				Process p = pb.start();
				System.out.printf("Calling octave to do CCA...."); System.out.flush();
				StopWatch stopwatch = new StopWatch(); 
				p.waitFor();
				System.out.printf("done [Time: %s]\n",stopwatch); System.out.flush();
				InputStream is = p.getInputStream();
				BufferedReader reader = new BufferedReader(new InputStreamReader(is));
				while (true) {
					String ln = reader.readLine();
					if (ln==null) { break; }
					System.out.println(ln);
				}
			} catch (InterruptedException inner) {
				inner.printStackTrace();
			}
			
						
			this.U = readMatrix(uFile);			
			this.V = readMatrix(vFile);
			this.iU = readMatrix(iuFile);			
			this.iV = readMatrix(ivFile);
			this.r = readColumnVector(rFile);
			System.out.println("r: " + Arrays.toString(r));
			
//			rowNormalize(U);
//			rowNormalize(V);

			
		} catch (Exception e) {			
			e.printStackTrace();
			System.err.println("Aborting CCA");
			return ;
		} 		
	}
	
	private void l2Normalize(double[] vec) {
		double sum = 0.0; 
		for (int i=0; i < vec.length; ++i) { sum += vec[i] * vec[i]; }
		double norm = Math.sqrt(sum);
		if (norm > 0.0) {
			for (int i=0; i < vec.length; ++i) { vec[i] /= norm; }
		}
	}
	
	private void rowNormalize(double[][] mat) {
		for (double[] row: mat) {
			l2Normalize(row); 
		}
	}
	
	private double[] readColumnVector(String path) {
		List<Double> nums = new ArrayList<Double>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				if (line.contains("#")) continue;
				String[] fields = line.trim().split("\\s+");				
				nums.add(Double.parseDouble(fields[0]));
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}		
		double[] arr = new double[nums.size()];
		for (int i=0; i < nums.size(); ++i) {
			arr[i] = nums.get(i);
		}
		return arr;
	}
	
	private double[][] readMatrix(String path) {
		List<double[]> rows = new ArrayList<double[]>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				}
				if (line.contains("#")) continue;
				String[] fields = line.trim().split("\\s+");
				double[] row  =  new double[fields.length];
				for (int i=0; i < row.length; ++i) {
					row[i] = Double.parseDouble(fields[i]);
				}
				rows.add(row);
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}		
		double[][] mat = new double[rows.size()][];
		for (int i=0; i < rows.size(); ++i) {
			mat[i] = rows.get(i);
		}
		return mat;
	}

	private void writeAsciiMatrix(double[][] mat, String path) {
		try {
			BufferedWriter br = new BufferedWriter(new FileWriter(path)) ;
			for (double[] row: mat) {
				StringBuilder sb = new StringBuilder();
				for (double elem: row) sb.append("\t"+elem);
				br.write(sb.toString()+"\n");
			}
			br.flush(); br.close();
			br.close();
		} catch (IOException e) {			
			e.printStackTrace();
		}
	}
	
	/**
	 * Ignore components that fall below <code>thresh</code>
	 * @param thresh
	 */
	public void setCorrelationThreshold(double thresh) {
		this.thresh = thresh;
		throw new UnsupportedOperationException();
	}
	
	private int maxDimen = Integer.MAX_VALUE;
	
	public void setMaxProjectedDimension(int maxDimen) {
		this.maxDimen = maxDimen;
	}
		
	private double[][] U;
	private double[][] V;
	private double[][] iU;
	private double[][] iV;
	private double[] r;
	private double thresh = 0.5;
	
	public double[][] getU() {
		return U;
	}
	
	public double[][] getV() {
		return V;
	}
	
	public static void main(String[] args) {
		int n = args.length > 0 ? Integer.parseInt(args[0]) : 100;
		double[][] mat = new double[n][n];
		for (int i=0; i < n; ++i) { mat[i][i] = 1.0; }		
		CCA cca = new CCA();
		cca.doCCA(mat, mat);
		cca.setMaxProjectedDimension(1);
		Pair<double[],double[]> leftProj = cca.getProjectedLeftVector(mat[0]);
		Pair<double[],double[]> rightProj = cca.getProjectedRightVector(mat[0]);
		System.out.println("left: " + Arrays.toString(leftProj.getFirst()));
		System.out.println("ileft: " + Arrays.toString(leftProj.getSecond()));
		System.out.println("right: " + Arrays.toString(rightProj.getFirst()));
		System.out.println("iright: " + Arrays.toString(rightProj.getSecond()));
	}
	
	public Pair<double[], double[]> getProjectedLeftVector(double[] elem) {
		List<Double> realProj = new ArrayList<Double>();
		List<Double> imagProj = new ArrayList<Double>();
		for (int i=0; i < U.length && i < maxDimen; ++i) {
			double rVal = r[i];
			if (rVal <= thresh) {
				break;
			}
			if (useComplexProjections) {
				Pair<Double,Double> val = complexInnerProduct(Pair.newPair(elem, new double[elem.length]), Pair.newPair(U[i],iU[i]));
				realProj.add(val.getFirst());
				imagProj.add(val.getSecond());
			} else {
				realProj.add(DoubleArrays.innerProduct(elem, U[i]));
			}
		}
		
		double[] realArr = new double[realProj.size()];
		double[] imagArr = new double[imagProj.size()];
		for (int i=0; i < realProj.size(); ++i) {
			realArr[i] = realProj.get(i);
		}
		for (int i=0; i < imagProj.size(); ++i) {
			imagArr[i] = imagProj.get(i);
		}
		
		l2Normalize(realArr);
//		l2Normalize(imagArr);
		
		Pair<double[],double[]> proj = Pair.newPair(realArr, imagArr);
		
		
		return proj;
	}
	
	public Pair<double[], double[]> getProjectedRightVector(double[] elem) {
		List<Double> realProj = new ArrayList<Double>();
		List<Double> imagProj = new ArrayList<Double>();
		for (int i=0; i < V.length && i < maxDimen; ++i) {
			double rVal = r[i];
			if (rVal <= thresh) {
				break;
			}
			if (useComplexProjections) {
				Pair<Double,Double> val = complexInnerProduct(Pair.newPair(elem, new double[elem.length]), Pair.newPair(V[i],iV[i]));
				realProj.add(val.getFirst());
				imagProj.add(val.getSecond());
			} else {
				realProj.add(DoubleArrays.innerProduct(elem, V[i]));
			}
		}
		
		double[] realArr = new double[realProj.size()];
		double[] imagArr = new double[imagProj.size()];
		for (int i=0; i < realProj.size(); ++i) {
			realArr[i] = realProj.get(i);
		}
		for (int i=0; i < imagProj.size(); ++i) {
			imagArr[i] = imagProj.get(i);
		}
		l2Normalize(realArr);
		Pair<double[],double[]> proj = Pair.newPair(realArr, imagArr);
		
		// why do we do this?
		//l2Normalize(arr);
		
		return proj;
	}
	
	// complex arith
	
	public static Pair<Double,Double> complexAdd(Pair<Double,Double> x, Pair<Double,Double> y, double scalar) {
		Pair<Double,Double> sum = new Pair<Double, Double>(x.getFirst()+(scalar*y.getFirst()),x.getSecond()+(scalar*y.getSecond()));
		return sum;
	}
	
	public static Pair<Double,Double> complexAddInPlace(Pair<Double,Double> x, Pair<Double,Double> y, double scalar) {
		x.setFirst(x.getFirst()+(scalar*y.getFirst()));
		x.setSecond(x.getSecond()+(scalar*y.getSecond()));
		return x;
	}
	
	public static Pair<Double,Double> complexMult(Pair<Double,Double> x, Pair<Double,Double> y) {
		Pair<Double,Double> prod = new Pair<Double, Double>((x.getFirst()*y.getFirst() - x.getSecond()*y.getSecond()), (x.getFirst()*y.getSecond() + y.getFirst()*x.getSecond()));
		return prod;
	}
	
	public static Pair<Double,Double> complexDiv(Pair<Double,Double> x, Pair<Double,Double> y) {
		double denom = Math.pow(y.getFirst(),2)+Math.pow(y.getSecond(), 2);
		Pair<Double,Double> div = new Pair<Double, Double>((x.getFirst()*y.getFirst()+x.getSecond()*y.getSecond())/denom,
				(x.getSecond()*y.getFirst()-x.getFirst()*y.getSecond())/denom);
		return div;
	}
	
	public static double complexNorm(Pair<Double,Double> x) {
		return Math.sqrt(Math.pow(x.getFirst(), 2) + Math.pow(x.getSecond(), 2));
	}
	
	public static Pair<Double,Double> complexInnerProduct(Pair<double[],double[]> x, Pair<double[],double[]> y) {
		if (x.getFirst().length != y.getFirst().length)
			throw new RuntimeException("diff lengths: " + x.getFirst().length + " "
					+ y.getFirst().length);
		Pair<Double, Double> result = new Pair<Double, Double>(0.0,0.0);
		for (int i = 0; i < x.getFirst().length; i++) {
			complexAddInPlace(result, complexMult(Pair.newPair(x.getFirst()[i], x.getSecond()[i]), Pair.newPair(y.getFirst()[i], y.getSecond()[i])), 1.0);
		}
		return result;
	}
	
}
