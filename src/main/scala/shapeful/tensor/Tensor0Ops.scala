package shapeful.tensor

import shapeful.tensor.Tensor.Tensor0

/** Methods that work only for scalar tensors **/
object Tensor0Ops {
  extension (t : Tensor0) 
    def item : Float = t.stensor.item
}
