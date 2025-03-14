package shapeful.tensor

import shapeful.tensor.Tensor.Tensor1
import scala.annotation.targetName
import shapeful.tensor.Tensor.Tensor2

object Tensor2Ops {
  
extension [A, B](tensor: Tensor[(A, B)])

  @targetName("matmulvec")
  def matmul(other: Tensor1[B]): Tensor[Tuple1[A]] = {
    val newt = tensor.stensor.matmul(other.stensor)
    val newShape = Shape[A](tensor.shape.dim[A])
    val t = new Tensor[Tuple1[A]](newShape, newt)
    t
  }

  @targetName("matmulmat")
  def matmul[C](other: Tensor2[B, C]): Tensor[(A, C)] = {
    val newt = tensor.stensor.matmul(other.stensor)
    val newShape = Shape[A, C](tensor.shape.dim[A], tensor.shape.dim[C])
    val t = new Tensor[(A, C)](newShape, newt)
    t
  }

   @targetName("add horizontal")
    def addAlongDim[Dim](other : Tensor1[Dim]) : Tensor[(A, B)] = {
        val newShape = tensor.shape.updateValue[Dim](other.shape.dim[Dim] + tensor.shape.dim[Dim])
        val newt = tensor.stensor.add(other.stensor)
        new Tensor[(A, B)](newShape, newt)
    }
}
