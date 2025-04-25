package shapeful.nn

import shapeful.tensor.{Tensor2, Tensor1}
import shapeful.tensor.Tensor2Ops.*
import torch.Float32
import shapeful.autodiff.Autodiff.jacobian
import shapeful.autodiff.Autodiff
import shapeful.autodiff.Params
import shapeful.tensor.Tensor3
import shapeful.tensor.Shape2
import shapeful.tensor.Variable2
import shapeful.tensor.Variable1
import torch.DType.float32
import shapeful.tensor.Tensor0
import shapeful.tensor.Shape3
import shapeful.tensor.Tensor4
import shapeful.tensor.Shape4
import shapeful.Label


trait Transformation4[From1 <: Label, From2 <: Label, From3 <: Label, To1 <: Label, To2 <: Label, To3 <: Label, DType <: torch.DType]:
    self =>
    def apply[Data <: Label](x : Tensor4[Data, From1, From2, From3, DType]) : Tensor4[Data, To1, To2, To3, DType]
    def andThen[OtherTo1 <: Label, OtherTo2 <: Label, OtherTo3 <: Label](other : Transformation4[To1, To2, To3, OtherTo1, OtherTo2, OtherTo3, DType]) : Transformation4[From1, From2, From3,  OtherTo1, OtherTo2, OtherTo3, DType] =
        new Transformation4[From1, From2, From3, OtherTo1, OtherTo2, OtherTo3, DType] {
            override def apply[Data <: Label](x : Tensor4[Data, From1, From2, From3, DType]) : Tensor4[Data, OtherTo1, OtherTo2, OtherTo3, DType] = other.apply(self.apply(x))
        }
    def andThen[OtherTo <: Label](other : Transformation4To2[To1, To2, To3, OtherTo, DType]) : Transformation4To2[From1, From2, From3,  OtherTo, DType] =
        new Transformation4To2[From1, From2, From3, OtherTo, DType] {
            override def apply[Data <: Label](x : Tensor4[Data, From1, From2, From3, DType]) : Tensor2[Data, OtherTo, DType] = other.apply(self.apply(x))
        }
object Transformation4:
    //Define operator with explicit precedence
    extension [From1 <: Label, From2 <: Label, From3 <: Label, To1 <: Label, To2 <: Label, To3 <: Label, DType <: torch.DType](
        self: Transformation4[From1, From2, From3, To1, To2, To3, DType])
        infix def |>[OtherTo1 <: Label, OtherTo2 <: Label, OtherTo3 <: Label](other: Transformation4[To1, To2, To3, OtherTo1, OtherTo2, OtherTo3, DType]):  Transformation4[From1, From2, From3, OtherTo1, OtherTo2, OtherTo3, DType] =
            self.andThen(other)
    
class Conv2D[FromChannel <: Label, From1 <: Label, From2 <: Label, ToChannel <: Label, To1 <: Label, To2 <: Label](
    weight : Tensor4[ToChannel, FromChannel, From1, From2,  Float32],
    bias : Tensor1[ToChannel, Float32],
    stride : Int = 1,
    padding : Int = 0
) extends Transformation4[FromChannel, From1, From2, ToChannel, To1, To2,  Float32]:
    self =>
    override def apply[Data <: Label](x : Tensor4[Data, FromChannel, From1, From2,  Float32]) : Tensor4[Data, ToChannel, To1, To2,  Float32] =

        val newT = torch.nn.functional.conv2d(
            x.repr.to(float32), weight.repr, bias.repr, stride = stride, padding = padding
        )


        val outHeight = ((x.shape.dim[From1] + 2*padding - weight.shape.dim[From1]) / stride) + 1
        val outWidth = ((x.shape.dim[From2] + 2*padding - weight.shape.dim[From2]) / stride) + 1
        
        val newShape = new Shape4[Data, ToChannel, To1, To2](
            x.shape.dim[Data],
            weight.shape.dim[ToChannel],
            outHeight,
            outWidth
        )

        new Tensor4[Data, ToChannel, To1, To2, Float32](
            newShape,
            newT,
            newT.dtype
        )
    





trait Transformation4To2[From1 <: Label, From2 <: Label, From3 <: Label, To <: Label, DType <: torch.DType]:
    self =>
    def apply[Data <: Label](x : Tensor4[Data, From1, From2, From3, DType]) : Tensor2[Data, To, DType]
    def andThen[OtherTo <: Label](other : Transformation2[To, OtherTo, DType]) : Transformation4To2[From1, From2, From3, OtherTo, DType] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] =
        new Transformation4To2[From1, From2, From3, OtherTo, DType] {
            override def apply[Data <: Label](x : Tensor4[Data, From1, From2, From3, DType]) : Tensor2[Data, OtherTo, DType] = other.apply(self.apply(x))
        }

object Transformation4To2:
    //Define operator with explicit precedence
    extension [From1 <: Label, From2 <: Label, From3 <: Label, To <: Label, DType <: torch.DType](self: Transformation4To2[From1, From2, From3, To, DType])
        infix def |>[OtherTo <: Label](other: Transformation2[To, OtherTo, DType]):  Transformation4To2[From1, From2, From3, OtherTo, DType] =
            self.andThen(other)

class Flatten4[From1 <: Label, From2 <: Label, From3 <: Label, To <: Label, DType <: torch.DType]() extends Transformation4To2[From1, From2, From3, To, DType]:
    self =>
    override def apply[Data <: Label](x : Tensor4[Data, From1, From2, From3, DType]) : Tensor2[Data, To, DType] = 
        val flattenedrepr = x.repr.reshape(x.shape.dim1, x.shape.dim2 * x.shape.dim3 * x.shape.dim4)
        new Tensor2[Data, To, DType](new Shape2(x.shape.dim1, x.shape.dim2 * x.shape.dim3 * x.shape.dim4), flattenedrepr, flattenedrepr.dtype)

class MaxPooling[FromChannel <: Label, From1 <: Label, From2 <: Label, ToChannel <: Label, To1 <: Label, To2 <: Label](
    kernelSize : Int = 2,
    stride : Int = 2,
    padding : Int = 0
) extends Transformation4[FromChannel, From1, From2, ToChannel, To1, To2, Float32]:
    self =>
    override def apply[Data <: Label](x : Tensor4[Data, FromChannel, From1, From2, Float32]) : Tensor4[Data, ToChannel, To1, To2, Float32] =
        val newT = torch.nn.functional.maxPool2d(x.repr.to(float32), kernelSize, stride = stride, padding = padding)
        val outHeight = ((x.shape.dim[From1] + 2*padding - kernelSize) / stride) + 1
        val outWidth = ((x.shape.dim[From2] + 2*padding - kernelSize) / stride) + 1
        
        val newShape = new Shape4[Data, ToChannel, To1, To2](
            x.shape.dim[Data],
            x.shape.dim[FromChannel],
            outHeight,
            outWidth
        )

        new Tensor4[Data, ToChannel, To1, To2, Float32](
            newShape,
            newT.to(float32),
            newT.dtype
        )