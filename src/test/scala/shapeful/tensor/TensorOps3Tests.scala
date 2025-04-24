package shapeful.tensor

import munit.FunSuite
import torch.Float32
import shapeful.tensor.TensorOps.{add}
import shapeful.tensor.Tensor2Ops.sum
import shapeful.tensor.Tensor3Ops.*
import shapeful.tensor.TensorOps.norm
import shapeful.tensor.TensorOps.sub


class TensorOps3Tests extends FunSuite {
    test("map over dim1") {
        val t = Tensor3.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 2, "dim3" ->> 2), Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))
        def add1(t : Tensor2["dim1", "dim2", Float32]) : Tensor2["dim1", "dim2", Float32] =  t.add(Tensor0(1f))
        val tnew = t.map["dim3"](add1)

        val tExpected = Tensor3.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 2, "dim3" ->> 2), Array(2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f))

        assertEqualsFloat(tnew.norm.sub(tExpected.norm).item, 0f, 0.001f)        
    }


    test("reduce over dim1") {
        val t = Tensor3.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 2, "dim3" ->> 2), Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))
        val tnew = t.reduce["dim1"](t => Tensor0(0f))
        val tExpected = Tensor1.fromSeq(Shape("dim1" ->> 2), Array(0f, 0f))

        assertEqualsFloat(tnew.norm.sub(tExpected.norm).item, 0f, 0.001f)
    }


}