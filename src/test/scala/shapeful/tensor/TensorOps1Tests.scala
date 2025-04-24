package shapeful.tensor

import munit.FunSuite
import torch.Float32
import shapeful.tensor.TensorOps.*
import shapeful.tensor.Tensor1Ops.*
import torch.Int32
import torch.DType.int32
import shapeful.autodiff.Params
import shapeful.autodiff.Autodiff

class TensorOps1Tests extends FunSuite {
   
    test("can map a float to an int tensor") {
        val t : Tensor1["dim1", Float32]= Tensor1.fromSeq(Shape("dim1" ->> 3), Array(0.1f, 1.6f, 2.3f))
        val tnew = t.map(t => Tensor0(t.item.toInt, int32))

        assertEquals(tnew(0).item, 0)
        assertEquals(tnew(1).item, 1)
        assertEquals(tnew(2).item, 2)
        
    }


    test("correctly applies the mapping function") {
        val t = Tensor1.fromSeq(Shape("dim1" ->>3), Array(1f, 2f, 3f))
        val tnew = t.map(t => t.mul(Tensor0(2f)))

        assertEqualsFloat(tnew(0).item, 2f, 0.001)
        assertEqualsFloat(tnew(1).item, 4f, 0.001)
        assertEqualsFloat(tnew(2).item, 6f, 0.001)
    }

    test("the gradient correctly propagates") {

        val x = Params(Map("x"->Tensor1.fromSeq(Shape("dim1" ->>2), Array(1f, 1f)).toVariable))
        val f = (params : Params) => 
            val x = params.get[Variable1["dim1"]]("x")
            val t = x.map(t => t.mul(Tensor0(2f)))
            t.sum

        val grad = Autodiff.deriv(f)(x).get[Tensor1["dim1", Float32]]("x")
        assertEqualsFloat(grad(0).item, 2f, 0.001)
        assertEqualsFloat(grad(1).item, 2f, 0.001)
    }

}