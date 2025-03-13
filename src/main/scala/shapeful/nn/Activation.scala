package shapeful.nn

import shapeful.tensor.Tensor.Tensor2
import shapeful.tensor.Tensor
import scala.compiletime.{constValue, erasedValue, summonFrom}

object Activation {

    inline def inlineIndexOf[Dims <: Tuple, A]: Int = inline erasedValue[Dims] match
        case _: (A *: _) => 0
        case _: (_ *: rest) => 1 + inlineIndexOf[rest, A]
        case _: EmptyTuple => -1
  
    inline def softmax[Dims <: Tuple, Dim](x: Tensor[Dims]): Tensor[Dims] =
        val i = inlineIndexOf[Dims, Dim]
        val newtensor = torch.softmax(x.stensor, dim = i, dtype = torch.float32)
        new Tensor[Dims](newtensor, x.shape.toList)

    inline def relu[Dims <: Tuple, Dim](x: Tensor[Dims]): Tensor[Dims] =
        val i = inlineIndexOf[Dims, Dim]
        val relu = torch.nn.ReLU(false)
        val newtensor = relu(x.stensor)
        new Tensor[Dims](newtensor, x.shape.toList)

}
