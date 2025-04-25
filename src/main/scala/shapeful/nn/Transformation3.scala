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
import shapeful.Label


trait Transformation3[From1 <: Label, From2 <: Label, To1 <: Label, To2 <: Label, DType <: torch.DType]:
    self =>
    def apply[Data <: Label](x : Tensor3[Data, From1, From2, DType]) : Tensor3[Data, To1, To2, DType]
    def andThen[OtherTo1 <: Label, OtherTo2 <: Label](other : Transformation3[To1, To2, OtherTo1, OtherTo2, DType]) : Transformation3[From1, From2, OtherTo1, OtherTo2, DType] =
        new Transformation3[From1, From2, OtherTo1, OtherTo2, DType] {
            override def apply[Data <: Label](x : Tensor3[Data, From1, From2, DType]) : Tensor3[Data, OtherTo1, OtherTo2, DType] = other.apply(self.apply(x))
        }
    def andThen[OtherTo <: Label](other : Transformation3To2[To1, To2, OtherTo, DType]) : Transformation3To2[From1, From2, OtherTo, DType] =
        new Transformation3To2[From1, From2, OtherTo, DType] {
            override def apply[Data <: Label](x : Tensor3[Data, From1, From2, DType]) : Tensor2[Data, OtherTo, DType] = other.apply(self.apply(x))
        }

object Transformation3:
    //Define operator with explicit precedence
    extension [From1 <: Label, From2 <: Label, To1 <: Label, To2 <: Label, DType <: torch.DType](
        self: Transformation3[From1, From2, To1, To2, DType])
        infix def |>[OtherTo1 <: Label, OtherTo2 <: Label](other: Transformation3[To1, To2, OtherTo1, OtherTo2, DType]):  Transformation3[From1, From2, OtherTo1, OtherTo2, DType] =
            self.andThen(other)
    extension [From1 <: Label, From2 <: Label, To1 <: Label, To2 <: Label, DType <: torch.DType](
        self: Transformation3[From1, From2, To1, To2, DType])
        infix def |>[OtherTo <: Label](other: Transformation3To2[To1, To2, OtherTo, DType]):  Transformation3To2[From1, From2, OtherTo, DType] =
            self.andThen(other)





///////////////////////////////
// Transformation3To2
//////////////////////////////

/**
 * A feature transform. Takes a bunch of data with features of type From and transforms them to features of type To.
 */
trait Transformation3To2[From1 <: Label, From2 <: Label, To <: Label, DType <: torch.DType]:
    self =>
    def apply[Data <: Label](x : Tensor3[Data, From1, From2, DType]) : Tensor2[Data, To, DType]
    def andThen[OtherTo <: Label](other : Transformation2[To, OtherTo, DType]) : Transformation3To2[From1, From2, OtherTo, DType] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] =
        new Transformation3To2[From1, From2, OtherTo, DType] {
            override def apply[Data <: Label](x : Tensor3[Data, From1, From2, DType]) : Tensor2[Data, OtherTo, DType] = other.apply(self.apply(x))
        }

object Transformation3To2:
    //Define operator with explicit precedence
    extension [From1 <: Label, From2 <: Label, To <: Label, DType <: torch.DType](self: Transformation3To2[From1, From2, To, DType])
        infix def |>[OtherTo <: Label](other: Transformation2[To, OtherTo, DType]):  Transformation3To2[From1, From2, OtherTo, DType] =
            self.andThen(other)

class Flatten3[From1 <: Label, From2 <: Label, To <: Label, DType <: torch.DType]() extends Transformation3To2[From1, From2, To, DType]:
    self =>
    override def apply[Data <: Label](x : Tensor3[Data, From1, From2, DType]) : Tensor2[Data, To, DType] = 
        val flattenedrepr = x.repr.reshape(x.shape.dim1, x.shape.dim2 * x.shape.dim3)
        new Tensor2[Data, To, DType](new Shape2(x.shape.dim1 * x.shape.dim2, x.shape.dim3), flattenedrepr, flattenedrepr.dtype)
        



