package kernelcca;

import edu.berkeley.nlp.util.StringUtils;
import org.junit.Test;

/**
 * Created with IntelliJ IDEA.
 * User: Tomer
 * Date: 4/8/13
 * Time: 4:37 PM
 * To change this template use File | Settings | File Templates.
 */
public class IncompleteCholeskyDecompositionTest {
    @Test
    public void testGetRepn() throws Exception {
        LinearKernel K = new LinearKernel();
        double[][] data = new double[][] {{1,3},{1,4},{1,-1}};

        double eta = 0.01;
        IncompleteCholeskyDecomposition icd = new IncompleteCholeskyDecomposition(K, data, eta);

        double[] z = new double[] {3,-2};
        double[] r = icd.getRepn(z);
        double[] nu = icd.getNu();
        int[] perm = icd.getPerm();
        double[][] R = icd.getR();


        System.out.print("r = ");
        printDoubleArray(r);
        System.out.println("eta = "+eta + "\n");
        System.out.print("nu = ");
        printDoubleArray(nu);
        System.out.print("perm = ");
        printIntArray(perm);
        System.out.println("R =");
        printDoubleArray2(R);
    }

    public static void printDoubleArray2(double[][] arr) {
        int n = arr.length;
        for (int i=0; i<n;i++) {
            printDoubleArray(arr[i]);
        }
    }

    public static void printDoubleArray(double[] arr) {
        int n = arr.length;
        System.out.print(arr[0]);
        for (int i=1; i<n; i++) {
            System.out.print(" ");
            System.out.print(arr[i]);
        }
        System.out.println();
    }

    public static void printIntArray(int[] arr) {
        int n = arr.length;
        System.out.print(arr[0]);
        for (int i=1; i<n; i++) {
            System.out.print(" ");
            System.out.print(arr[i]);
        }
        System.out.println();
    }
}
