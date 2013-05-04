package kernelcca;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

class TypedPoint<T> {
  public enum PointType { X, Y };

  private PointType pointType;
  private T point;
  public TypedPoint(PointType pointType, T point) {
    this.pointType = pointType;
    this.point = point;
  }
  public PointType getPointType() { return pointType; }
  public T getPoint() { return point; }
  public static <T> TypedPoint<T> newX(T point) { return new TypedPoint(PointType.X, point); }
  public static <T> TypedPoint<T> newY(T point) { return new TypedPoint(PointType.Y, point); }
  public boolean isX() { return pointType == PointType.X; }
  public boolean isY() { return pointType == PointType.Y; }
}

class TypedKernel<T> implements Kernel<TypedPoint<T>> {
  Kernel<T> kx, ky, kc;
  public TypedKernel(Kernel<T> kx, Kernel<T> ky, Kernel<T> kc) {
    this.kx = kx; this.ky = ky; this.kc = kc;
  }
  public double dot(TypedPoint<T> a, TypedPoint<T> b) {
    // Choose which kernel to use depending on the types of a and b
    Kernel<T> k;
    if(a.isX()) k = b.isX() ? kx : kc;
    else        k = b.isX() ? kc : ky;
    return k.dot(a.getPoint(), b.getPoint());
  }
}
