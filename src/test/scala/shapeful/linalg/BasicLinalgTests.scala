package shapeful.tensor

import munit.FunSuite
import torch.Float32
import shapeful.tensor.TensorOps.add
import shapeful.tensor.Tensor2Ops.map
import shapeful.tensor.TensorOps.norm
import shapeful.tensor.TensorOps.sub
import shapeful.tensor.Tensor2Ops.inv
import shapeful.linalg.BasicLinalg
import shapeful.tensor.Tensor2Ops.matmul
import shapeful.tensor.Tensor2Ops.transpose


class BasicLinalgTests extends FunSuite {

    test("inverse applied to itself should be identity") {
        val t = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 2), Array(4f, 7f, 2f, 6f))

        val tnew = BasicLinalg.inverse(BasicLinalg.inverse(t))
        assertEqualsFloat(tnew.norm.sub(t.norm).item, 0f, 0.001f)
    }

    test("test cholesky") {
        val t = Tensor2.fromSeq(Shape("dim1" ->> 2, "dim2" ->> 2), Array(4f, 2f, 2f, 6f))
        val tnew = BasicLinalg.cholesky(t)
        val trec = tnew.matmul(tnew.transpose)
        assertEqualsFloat(trec.norm.sub(t.norm).item, 0f, 0.001f)

    }
}