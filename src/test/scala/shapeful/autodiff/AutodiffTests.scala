package shapeful.autodiff

import munit.FunSuite
import shapeful.tensor.Shape
import shapeful.tensor.Tensor0
import torch.Float32
import shapeful.tensor.TensorOps.pow
import shapeful.tensor.Variable0
import shapeful.tensor.TensorOps.*
import shapeful.tensor.Shape1
import shapeful.tensor.~>
import shapeful.tensor.Variable1
import shapeful.tensor.TensorOps.add
import shapeful.tensor.Tensor1Ops.sum
import shapeful.tensor.Tensor1

class AutodiffTests extends FunSuite {

  test("diff of quadratic function1 of a scalar") {
    
    def f(params : Params): Tensor0[Float32] =
        params.get[Variable0]("x").pow(2)
    val df = deriv(f)

    for i <- -10 until 10 do
        val p = Params(Map("x" -> Variable0(i)))
        val dx = df(p).get[Variable0]("x")
        val expected = 2f * i
        assertEqualsFloat(dx.item, expected, 0.001)
  }

  test("diff of function2 of two scalars") {

    def f(params : Params) : Tensor0[Float32] = 
        val x = params.get[Variable0]("x")
        val y = params.get[Variable0]("y")
        x.mul(y)

    val df = deriv(f)

    for i <- -10 until 10;
        j <- -10 until 10 do
        val params = Params(Map("x" -> Variable0(i), "y" -> Variable0(j)))
        val grad = df(params)
        val dx = grad.get[Variable0]("x")
        val dy = grad.get[Variable0]("y")

        val expectedX = j.toFloat
        val expectedY = i.toFloat

        assertEqualsFloat(dx.item, expectedX, 0.001)
        assertEqualsFloat(dy.item, expectedY, 0.001)
  }

    test("diff of function2 of two vectors") {

        val shape = Shape("Data" ~> 2)

        def f(params : Params) : Tensor0[Float32] = 
            val x = params.get[Variable1["Data"]]("x")
            val y = params.get[Variable1["Data"]]("y")
            val m = x.add(y.mul(Tensor0(2f)))
            m.sum

        val df = deriv(f)

        val params = Params(Map(
            "x" -> Tensor1.fromSeq(shape, Seq(1,1)).toVariable, 
            "y" -> Tensor1.fromSeq(shape, Seq(2, 2)).toVariable)
        )

        val grad = df(params)
        val dx = grad.get[Variable1["Data"]]("x")
        val dy = grad.get[Variable1["Data"]]("y")

        val (expectedx1, expectedx2) = (1f, 1f)
        val (expectedy1, expectedy2) = (2f, 2f)

        assertEqualsFloat(dx(0).item, expectedx1, 0.001)
        assertEqualsFloat(dx(1).item, expectedx2, 0.001)

        assertEqualsFloat(dy(0).item, expectedy1, 0.001)
        assertEqualsFloat(dy(1).item, expectedy2, 0.001)
  }
}
