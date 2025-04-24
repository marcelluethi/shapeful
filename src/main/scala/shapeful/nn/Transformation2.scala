package shapeful.nn

import shapeful.tensor.{Tensor2, Tensor1}
import shapeful.tensor.Tensor2Ops.*
import torch.Float32
import shapeful.autodiff.Autodiff.jacobian
import shapeful.autodiff.Autodiff
import shapeful.autodiff.Params


/**
 * A feature transform. Takes a bunch of data with features of type From and transforms them to features of type To.
 */
trait Transformation2[From <: Singleton, To <: Singleton, DType <: torch.DType]:
    self =>
    def apply[Data <: Singleton](x : Tensor2[Data, From, DType]) : Tensor2[Data, To, DType]
    def andThen[OtherTo <: Singleton](other : Transformation2[To, OtherTo, DType]) : Transformation2[From, OtherTo, DType] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] =
        new Transformation2[From, OtherTo, DType] {
            override def apply[Data <: Singleton](x : Tensor2[Data, From, DType]) :Tensor2[Data, OtherTo, DType] = other.apply(self.apply(x))
        }
object Transformation2:
    //Define operator with explicit precedence
    extension [From <: Singleton, To <: Singleton, DType <: torch.DType](self: Transformation2[From, To, DType])
        infix def |>[OtherTo <: Singleton](other: Transformation2[To, OtherTo, DType]):  Transformation2[From, OtherTo, DType] =
            self.andThen(other)

class AffineTransformation[From <: Singleton, To <: Singleton](w : Tensor2[From, To, Float32], b : Tensor1[To, Float32]) extends Transformation2[From, To, Float32]:
    self =>
    override def apply[Data <: Singleton](x : Tensor2[Data, From, Float32]) : Tensor2[Data, To, Float32] =
        x.matmul(w).addTensor1(b)




// trait Transformation3[From1, From2, To1, To2]:
//     self =>
//     def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor3[Data, To1, To2]
//     def andThen[OtherTo1, OtherTo2](other : Transformation3[To1, To2, OtherTo1, OtherTo2]) : Transformation3[From1, From2, OtherTo1, OtherTo2] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] =
//         new Transformation3[From1, From2, OtherTo1, OtherTo2] {
//             override def apply[Data](x : Tensor3[Data, From1, From2]) :Tensor3[Data, OtherTo1, OtherTo2] = other.apply(self.apply(x))
//         }

//     def andThen[OtherTo](other : Projection3To2[To1, To2, OtherTo]) : Projection3To2[From1, From2, OtherTo] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] =
//         new Projection3To2[From1, From2, OtherTo] {
//             override def apply[Data](x : Tensor3[Data, From1, From2]) :Tensor2[Data, OtherTo] = other.apply(self.apply(x))
//         }

// extension[From1, From2, To1, To2](self : Transformation3[From1, From2, To1, To2])
//     infix def |>[OtherTo1, OtherTo2](other: Transformation3[To1, To2, OtherTo1, OtherTo2]):  Transformation3[From1, From2, OtherTo1, OtherTo2] =
//         self.andThen(other)

// extension[From1, From2, To1, To2](self : Transformation3[From1, From2, To1, To2])
//     infix def |>[OtherTo](other: Projection3To2[To1, To2, OtherTo]):  Projection3To2[From1, From2, OtherTo] =
//         self.andThen(other)


// trait Projection3To2[From1, From2, To]:
//     self =>
//     def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor2[Data, To]
//     def andThen[OtherTo](other : Transformation2[To, OtherTo]) : Projection3To2[From1, From2, OtherTo] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] =
//         new Projection3To2[From1, From2, OtherTo] {
//             override def apply[Data](x : Tensor3[Data, From1, From2]) :Tensor2[Data, OtherTo] = other.apply(self.apply(x))
//         }
// extension[From1, From2, To](self : Projection3To2[From1, From2, To])
//     infix def |>[OtherTo](other: Transformation2[To, OtherTo]):  Projection3To2[From1, From2, OtherTo] =
//         self.andThen(other)

// class Flatten[From1, From2, NewFrom] extends Projection3To2[From1, From2, NewFrom]:
//     self =>
//     def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor2[Data, NewFrom] =
//         x.reshape(Shape[Data, NewFrom](x.shape.dim[Data], x.shape.dim[From1] * x.shape.dim[From2]))

// class Conv2D[From1, From2, To1, To2, K1, K2](w : Tensor2[K1, K2]) extends Transformation3[From1, From2, To1, To2]:
//     self =>
//     def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor3[Data, To1, To2] =

//         // val xconv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
//         // val t = xconv(x.stensor.reshape(1, x.shape.dim[Data], 1, x.shape.dim[From1], x.shape.dim[From2]))
//         // new Tensor3(Shape[Data, To1, To2](t.shape(0), t.shape(1), t.shape(2)), t)
//         ???
