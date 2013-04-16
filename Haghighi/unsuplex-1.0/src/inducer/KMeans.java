/**
 * 
 */
package inducer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.math.DoubleArrays;
import fig.basic.Pair;

public class KMeans {
	int K ;
	List<double[]> datums ;
	int numTrys = 30;
	
	public KMeans(int K) {
		this.K = K;
	}
	
	public int[] cluster(List<double[]> datums0) {			
		datums = copy(datums0);
		
		List<double[]> bestCenters = null;
		double bestObjVal = Double.POSITIVE_INFINITY;
		
		for (int i=0; i < numTrys; ++i) {
			Collections.shuffle(datums);
			List<double[]> initCenters = datums.subList(0, K);				
			Pair<Double, List<double[]>> p = clusterTry(initCenters);
			if (p.getFirst() < bestObjVal) {
				bestObjVal = p.getFirst();
				bestCenters = p.getSecond();
			}
		}
		
		int[] clusterIds = new int[datums0.size()];
		for (int i=0; i < datums0.size(); ++i) {
			int k = findClosest(datums.get(i), bestCenters);
			clusterIds[i] = k;
		}
		
		System.err.printf("Best Cluster Score: %.3f\n",bestObjVal);
		
		return clusterIds;
	}
	
	private int findClosest(double[] x, List<double[]> centroids) {
		double minDist = Double.POSITIVE_INFINITY;
		int argMin = -1;
		for (int k=0; k < K; ++k) {
			double d = dist(x,centroids.get(k));
			if (d < minDist) {
				minDist = d;
				argMin = k;
			}
		}
		return argMin;
	}
	
	private double dist(double[] x, double[] y) {
		double sum =0.0;
		for (int i=0; i < x.length; ++i) {
			sum += (x[i] - y[i]) * (x[i] - y[i]);
		}
		return sum;
	}
	
	private List<double[]> copy(List<double[]> lst) {
		List<double[]> cpyList = new ArrayList<double[]>();
		for (double[] x: lst) { 
			double[] xCpy = new double[x.length];
			System.arraycopy(x, 0, xCpy, 0, x.length);
			cpyList.add(xCpy);
		}
		return cpyList;
	}
		
	private double[] getCentroid(List<double[]> elems) {
		int n = elems.get(0).length;
		double[] centroid = new double[n];
		for (double[] elem: elems) {
			DoubleArrays.addInPlace(centroid, elem);
		}
		double scale = 1.0 / elems.size();
		DoubleArrays.scale(centroid, scale);
		return centroid;
	}
	
	private Pair<Double, List<double[]>> clusterTry(List<double[]> initialCenters) {
		List<double[]> clusterCenters = copy(initialCenters);
		double oldObjValue = Double.POSITIVE_INFINITY;
		double objValue = 0.0;
		for (int i=0; i < 10; ++i) {
			
			List<List<double[]>> clusterMembers = new ArrayList<List<double[]>>();
			for(int k=0; k < K; ++k) { clusterMembers.add(new ArrayList<double[]>()); }								
			// Assign to Clusters 
			for (double[] datum: datums) {
				int closest = findClosest(datum, clusterCenters);
				clusterMembers.get(closest).add(datum);					 
			}
			// Find new centroids
			objValue = 0.0;
			for (int k=0; k < K; ++k) {					
				List<double[]> members = clusterMembers.get(k);
				double[] newCentroid = getCentroid(members);
				for (double[] member: members) {
					objValue += 0.5 * dist(newCentroid, member);
				}
				clusterCenters.set(k, newCentroid);
			}				
			if (Math.abs(oldObjValue-objValue) < 1.0e-4) {
				break;
			}
		}						
		return Pair.newPair(objValue, clusterCenters);
	}
	
	
}