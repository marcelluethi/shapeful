package shapeful.tensor

import munit.FunSuite
import torch.Float32
import shapeful.tensor.TensorOps.add
import shapeful.tensor.Tensor2Ops.map
import shapeful.tensor.TensorOps.norm
import shapeful.tensor.TensorOps.sub
import shapeful.tensor.Tensor2Ops.inv
import shapeful.tensor.Tensor2Ops.transpose
import shapeful.tensor.Tensor2Ops.reduce
import shapeful.tensor.Tensor1Ops.sum


class TensorOps2Tests extends FunSuite {
    test("map over dim1") {
        val t = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 3), Array(1f, 2f, 3f, 4f, 5f, 6f))
        def add1(t : Tensor1["dim2", Float32]) : Tensor1["dim2", Float32] =  t.add(Tensor0(1f))
        val tnew = t.map["dim1"](add1)

        val tExpected = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 3), Array(2f, 3f, 4f, 5f, 6f, 7f))

        assertEqualsFloat(tnew.norm.sub(tExpected.norm).item, 0f, 0.001f)
        
    }

    test("map over dim2") {
        val t = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 3), Array(1f, 2f, 3f, 4f, 5f, 6f))
        def add1(t : Tensor1["dim1", Float32]) : Tensor1["dim1", Float32] =  t.add(Tensor0(1f))
        val tnew = t.map["dim2"](add1)

        val tExpected = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 3), Array(2f, 3f, 4f, 5f, 6f, 7f))

        assertEqualsFloat(tnew.norm.sub(tExpected.norm).item, 0f, 0.001f)
        
    }

    test("reduce over dim1") {
        val t = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 3), Array(1f, 2f, 3f, 4f, 5f, 6f))
        val tnew = t.reduce["dim1"](t => t.sum)

        val tExpected = Tensor1.fromSeq(Shape("dim2" ->> 2), Array(9f, 12f))

        assertEqualsFloat(tnew.norm.sub(tExpected.norm).item, 0f, 0.001f)
    }

    test("transpose correctly swaps columns and rows") {
        val t = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 3), Array(1f, 2f, 3f, 4f, 5f, 6f))
        val tnew = t.transpose
        val tExpected = Tensor2.fromSeq(Shape("dim2" ->> 3, "dim1" ->> 2), Array(1f, 4f, 2f, 5f, 3f, 6f))

        assertEqualsFloat(tnew.norm.sub(tExpected.norm).item, 0f, 0.001f)
    }


}