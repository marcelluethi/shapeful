package shapeful.tensor

import shapeful.tensor.Tensor.Tensor0
import scala.annotation.targetName

/** operations that work tensors of all shapes **/
object TensorOps {
    
  extension[Dims <: Tuple] (tensor: Tensor[Dims] )
    @targetName("mul scalar")
    def mul(s : Tensor0) : Tensor[Dims] =
        val newt = tensor.stensor.mul(s.stensor)
        new Tensor[Dims](newt, newt.shape.toList)

    @targetName("mul tensor")
    def mul(s : Tensor[Dims]) : Tensor[Dims] =
        val newt = tensor.stensor.mul(s.stensor)
        new Tensor[Dims](newt, newt.shape.toList)

    @targetName("add tensor")
    def add(other : Tensor[Dims]) : Tensor[Dims] = {
        val newt = tensor.stensor.add(other.stensor)
        new Tensor[Dims](newt, newt.shape.toList)
    }

    @targetName("divtensor")
    def div(other : Tensor[Dims]) : Tensor[Dims] =
        val newt = tensor.stensor.div(other.stensor)
        new Tensor[Dims](newt, newt.shape.toList)

    @targetName("divscalar")
    def div(scalar : Tensor0) : Tensor[Dims] =
        val newt = tensor.stensor.div(scalar.stensor)
        new Tensor[Dims](newt, newt.shape.toList)

    @targetName("add scalar")
    def add(other: Tensor0): Tensor[Dims] = {
        val newt = tensor.stensor.add(other.stensor)
        new Tensor[Dims](newt, newt.shape.toList)
    }
    @targetName("subtract tensor")
    def sub(other: Tensor[Dims]): Tensor[Dims] = {
        val newt = tensor.stensor.sub(other.stensor)
        new Tensor[Dims](newt, newt.shape.toList)
    }

    def pow(exp : Int) : Tensor[Dims] =
        val newt = tensor.stensor.pow(exp)
        new Tensor[Dims] (newt, newt.shape.toList)

    def log: Tensor[Dims] =
        val newt = tensor.stensor.log
        new Tensor[Dims](newt, newt.shape.toList)

    def clamp(min : Option[Float], max : Option[Float]): Tensor[Dims] =

        val newt = torch.clamp(tensor.stensor, min, max)
        new Tensor[Dims](newt, newt.shape.toList)
}
