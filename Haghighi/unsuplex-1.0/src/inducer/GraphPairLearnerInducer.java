package inducer;

import Jama.Matrix;
import fig.basic.Indexer;
import kernelcca.Kernel;

/**
 * Created with IntelliJ IDEA.
 * User: Tomer
 * Date: 4/9/13
 * Time: 10:58 AM
 * To change this template use File | Settings | File Templates.
 */
public class GraphPairLearnerInducer extends PairLearnerInducer {
    private final Matrix domG;
    private final Matrix codomG;
    private final double lambda;

    public GraphPairLearnerInducer(Kernel<String> domKernel, Kernel<String> codomKernel, Matrix domG, Matrix codomG, double lambda) {
        super(domKernel, codomKernel);
        this.domG = domG;
        this.codomG = codomG;
        this.lambda = lambda;
    }

    @Override
    public double[][] getMatchingMatrix(Indexer<String> domWords, Indexer<String> codomWords) {
        double[][] M = super.getMatchingMatrix(domWords, codomWords);    //To change body of overridden methods use File | Settings | File Templates.
        Matrix M0 = new Matrix(M);
        Matrix M1 = domG.times(M0).times(codomG);
        M0 = M0.plus(M1.times(lambda));
        return M0.getArray();
    }
}
