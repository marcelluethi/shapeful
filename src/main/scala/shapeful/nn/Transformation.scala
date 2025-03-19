package shapeful.nn

import shapeful.tensor.Tensor.Tensor2
import shapeful.tensor.Tensor.Tensor1
import shapeful.tensor.Tensor2Ops.matmul
import shapeful.tensor.Tensor2Ops.addAlongDim


/**
 * A feature transform. Takes a bunch of data with features of type From and transforms them to features of type To.
 */
trait Transformation[From, To]:
    self => 
    def apply[Data](x : Tensor2[Data, From]) : Tensor2[Data, To] 
    def andThen[OtherTo](other : Transformation[To, OtherTo]) : Transformation[From, OtherTo] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] = 
        new Transformation[From, OtherTo] {
            override def apply[Data](x : Tensor2[Data, From]) :Tensor2[Data, OtherTo] = other.apply(self.apply(x))
        }

//Define operator with explicit precedence
extension [From, To](self: Transformation[From, To])
    infix def |>[OtherTo](other: Transformation[To, OtherTo]):  Transformation[From, OtherTo] = 
        self.andThen(other)

class AffineTransformation[From, To](w : Tensor2[From, To], b : Tensor1[To]) extends Transformation[From, To]: 
    self => 
    def apply[Data](x : Tensor2[Data, From]) : Tensor2[Data, To] =
        x.matmul(w).addAlongDim(b)

// class Flatten[From, To]() extends Transformation[From, To]: 
//     self => 
//     def apply[Data](x : Tensor2[Data, From]) : Tensor2[Data, To] =
//         x.reshape(shape)

// class Conv2D[From, To](w : Tensor2[From, To], b : Tensor1[To]) extends Transformation[From, To]: 
//     self => 
//     def apply[Data](x : Tensor2[Data, From]) : Tensor2[Data, To] =

//         torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)