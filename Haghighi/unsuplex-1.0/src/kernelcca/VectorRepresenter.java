package kernelcca;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

/**
 * Purpose: convert a data point of type T into a vector.
 */
public interface VectorRepresenter<T> {
  public int getD();
  public double[] getRepn(int i); // Get representation of the i-th data point
  public double[] getRepn(T x);
}

class ShiftVectorRepresenter<T> implements VectorRepresenter<T> {
  VectorRepresenter<T> vr;
  int shift;
  public ShiftVectorRepresenter(VectorRepresenter<T> vr, int shift) {
    this.vr = vr;
    this.shift = shift;
  }
  public int getD() { return vr.getD(); }
  public double[] getRepn(int i) { return vr.getRepn(i+shift); }
  public double[] getRepn(T x) { return vr.getRepn(x); }
}
