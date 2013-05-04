package canco;

import java.util.Random;

public class BipartiteGraphMatcherTester {
	public static void main(String[] args) {
		int n = Integer.parseInt(args[0]);
		Random rand = new Random();
		System.out.printf("Starting graph matching test with n=%d\n",n);
		long start = System.currentTimeMillis();
		BipartiteMatcher matcher = new BipartiteMatcher(n);
		for (int i=0; i < n; ++i) {
			for (int j=0; j < n; ++j) {
				matcher.setWeight(i, j, rand.nextDouble());
			}
		}
		int[] matching = matcher.getMatching();
		long end = System.currentTimeMillis();
		double secs = (end-start) / 1.000;
		System.out.printf("With n=%d took %.3f seconds\n",n,secs);
	}
}
