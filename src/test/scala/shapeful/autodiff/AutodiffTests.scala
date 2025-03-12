package shapeful.autodiff

import munit.FunSuite
import shapeful.tensor.Tensor.Tensor0
import shapeful.tensor.Tensor
import shapeful.tensor.TensorOps.* 
import shapeful.tensor.Tensor0Ops.*
import shapeful.tensor.Tensor.Tensor1

class AutodiffTests extends FunSuite {


  test("diff of quadratic function1 of a scalar") {

    val f = (x : Tensor0) => x.pow(2) 
    val df = Autodiff.deriv(f)

    for i <- -10 until 10 do
        val x : Tensor0 = Tensor(i.toFloat, requiresGrad = true)
        val dx = df(Tuple1(x)).head
        val expected = 2f * i
        assertEqualsFloat(dx.item, expected, 0.001)
  }

  
  test("diff of function2 of two scalars") {

    val f = (x : Tensor0, y : Tensor0) => x.mul(y)

    val df = Autodiff.deriv(f)

    for i <- -10 until 10; 
        j <- -10 until 10 do
        val x : Tensor0 = Tensor(i.toFloat, requiresGrad = true)
        val y : Tensor0 = Tensor(j.toFloat, requiresGrad = true)
        val (dx, dy) = df((x, y))
        
        val expectedX = j.toFloat
        val expectedY = i.toFloat

        assertEqualsFloat(dx.item, expectedX, 0.001)
        assertEqualsFloat(dy.item, expectedY, 0.001)
  }

    test("diff of function2 of two vectors") {

        type Data = "data"
        given shapeful.tensor.Dimension[Data] = shapeful.tensor.Dimension.Symbolic[Data](2)
        
        val f = (x : Tensor1[Data], y : Tensor1[Data]) => 
            val m = x.add(y.mul(Tensor(2f, requiresGrad = false)))
            m.sum[Data]

        val df = Autodiff.deriv(f)


        val (dx, dy) = df((Tensor.fromSeq(Seq(1,1), requiresGrad = true), Tensor.fromSeq(Seq(2,2), requiresGrad = true)))

        val (expectedx1, expectedx2) = (1f, 1f)
        val (expectedy1, expectedy2) = (2f, 2f)

        assertEqualsFloat(dx[Data](0).item, expectedx1, 0.001)
        assertEqualsFloat(dx[Data](0).item, expectedx2, 0.001)

        assertEqualsFloat(dy[Data](0).item, expectedy1, 0.001)
        assertEqualsFloat(dy[Data](0).item, expectedy2, 0.001)
  }
}
