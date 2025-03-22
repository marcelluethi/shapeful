package shapeful.nn

import shapeful.tensor.Tensor.Tensor2
import shapeful.tensor.Tensor.Tensor1
import shapeful.tensor.Tensor2Ops.matmul
import shapeful.tensor.Tensor2Ops.addAlongDim
import shapeful.tensor.Tensor.Tensor3
import shapeful.tensor.Shape


/**
 * A feature transform. Takes a bunch of data with features of type From and transforms them to features of type To.
 */
trait Transformation2[From, To]:
    self => 
    def apply[Data](x : Tensor2[Data, From]) : Tensor2[Data, To] 
    def andThen[OtherTo](other : Transformation2[To, OtherTo]) : Transformation2[From, OtherTo] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] = 
        new Transformation2[From, OtherTo] {
            override def apply[Data](x : Tensor2[Data, From]) :Tensor2[Data, OtherTo] = other.apply(self.apply(x))
        }



//Define operator with explicit precedence
extension [From, To](self: Transformation2[From, To])
    infix def |>[OtherTo](other: Transformation2[To, OtherTo]):  Transformation2[From, OtherTo] = 
        self.andThen(other)


trait Transformation3[From1, From2, To1, To2]:
    self => 
    def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor3[Data, To1, To2] 
    def andThen[OtherTo1, OtherTo2](other : Transformation3[To1, To2, OtherTo1, OtherTo2]) : Transformation3[From1, From2, OtherTo1, OtherTo2] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] = 
        new Transformation3[From1, From2, OtherTo1, OtherTo2] {
            override def apply[Data](x : Tensor3[Data, From1, From2]) :Tensor3[Data, OtherTo1, OtherTo2] = other.apply(self.apply(x))
        }

    def andThen[OtherTo](other : Projection3To2[To1, To2, OtherTo]) : Projection3To2[From1, From2, OtherTo] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] = 
        new Projection3To2[From1, From2, OtherTo] {
            override def apply[Data](x : Tensor3[Data, From1, From2]) :Tensor2[Data, OtherTo] = other.apply(self.apply(x))
        }

extension[From1, From2, To1, To2](self : Transformation3[From1, From2, To1, To2])
    infix def |>[OtherTo1, OtherTo2](other: Transformation3[To1, To2, OtherTo1, OtherTo2]):  Transformation3[From1, From2, OtherTo1, OtherTo2] = 
        self.andThen(other)
        
extension[From1, From2, To1, To2](self : Transformation3[From1, From2, To1, To2])
    infix def |>[OtherTo](other: Projection3To2[To1, To2, OtherTo]):  Projection3To2[From1, From2, OtherTo] = 
        self.andThen(other)
        



class AffineTransformation[From, To](w : Tensor2[From, To], b : Tensor1[To]) extends Transformation2[From, To]: 
    self => 
    def apply[Data](x : Tensor2[Data, From]) : Tensor2[Data, To] =
        x.matmul(w).addAlongDim(b)

trait Projection3To2[From1, From2, To]:
    self => 
    def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor2[Data, To] 
    def andThen[OtherTo](other : Transformation2[To, OtherTo]) : Projection3To2[From1, From2, OtherTo] =//Tensor2[Data, From] => Tensor2[Data, OtherTo] = 
        new Projection3To2[From1, From2, OtherTo] {
            override def apply[Data](x : Tensor3[Data, From1, From2]) :Tensor2[Data, OtherTo] = other.apply(self.apply(x))
        }
extension[From1, From2, To](self : Projection3To2[From1, From2, To])
    infix def |>[OtherTo](other: Transformation2[To, OtherTo]):  Projection3To2[From1, From2, OtherTo] = 
        self.andThen(other)
        



class Flatten[From1, From2, NewFrom] extends Projection3To2[From1, From2, NewFrom]: 
    self => 
    def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor2[Data, NewFrom] =
        x.reshape(Shape[Data, NewFrom](x.shape.dim[Data], x.shape.dim[From1] * x.shape.dim[From2]))



class Conv2D[From1, From2, To1, To2, K1, K2](w : Tensor2[K1, K2]) extends Transformation3[From1, From2, To1, To2]: 
    self => 
    def apply[Data](x : Tensor3[Data, From1, From2]) : Tensor3[Data, To1, To2] =

        // val xconv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        // val t = xconv(x.stensor.reshape(1, x.shape.dim[Data], 1, x.shape.dim[From1], x.shape.dim[From2]))
        // new Tensor3(Shape[Data, To1, To2](t.shape(0), t.shape(1), t.shape(2)), t)
        ???