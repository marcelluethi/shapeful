package shapeful.tensor

import shapeful.tensor.Tensor.Tensor1
import scala.annotation.targetName

object Tensor2Ops {
  
extension [A, B](tensor: Tensor[(A, B)])

  @targetName("matmulvec")
  def matmul(other: Tensor1[B]): Tensor[Tuple1[A]] = {
    val newt = tensor.stensor.matmul(other.stensor)
    val t = new Tensor[Tuple1[A]](newt, newt.shape.toList)
    t
  }

  @targetName("matmulmat")
  def matmul[C](other: Tensor[(B, C)]): Tensor[(A, C)] = {
    val newt = tensor.stensor.matmul(other.stensor)
    val t = new Tensor[(A, C)](newt, newt.shape.toList)
    t
  }
}
