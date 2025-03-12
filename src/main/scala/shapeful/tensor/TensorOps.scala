package shapeful.tensor

import shapeful.tensor.Tensor.Tensor0
import scala.annotation.targetName

/** operations that work tensors of all shapes **/
object TensorOps {
    
  extension[Shape <: Tuple] (tensor: Tensor[Shape] )
    @targetName("mul scalar")
    def mul(s : Tensor0) : Tensor[Shape] =
        val newt = tensor.stensor.mul(s.stensor)
        new Tensor[Shape](newt, newt.shape.toList)

    @targetName("add tensor")
    def add(other : Tensor[Shape]) : Tensor[Shape] = {
        val newt = tensor.stensor.add(other.stensor)
        new Tensor[Shape](newt, newt.shape.toList)
    }

    @targetName("divtensor")
    def div(other : Tensor[Shape]) : Tensor[Shape] =
        val newt = tensor.stensor.div(other.stensor)
        new Tensor[Shape](newt, newt.shape.toList)

    @targetName("divscalar")
    def div(scalar : Tensor0) : Tensor[Shape] =
        val newt = tensor.stensor.div(scalar.stensor)
        new Tensor[Shape](newt, newt.shape.toList)

    @targetName("add scalar")
    def add(other: Tensor0): Tensor[Shape] = {
        val newt = tensor.stensor.add(other.stensor)
        new Tensor[Shape](newt, newt.shape.toList)
    }
    @targetName("subtract tensor")
    def sub(other: Tensor[Shape]): Tensor[Shape] = {
        val newt = tensor.stensor.sub(other.stensor)
        new Tensor[Shape](newt, newt.shape.toList)
    }

    def pow(exp : Int) : Tensor[Shape] =
        val newt = tensor.stensor.pow(exp)
        new Tensor[Shape] (newt, newt.shape.toList)

    def log: Tensor[Shape] =
        val newt = tensor.stensor.log
        new Tensor[Shape](newt, newt.shape.toList)


}
