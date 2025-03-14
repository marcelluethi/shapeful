package shapeful.tensor

import munit.FunSuite
import shapeful.tensor.Tensor.Tensor0
import shapeful.tensor.Tensor
import shapeful.tensor.TensorOps.* 
import shapeful.tensor.Tensor0Ops.*
import shapeful.tensor.Tensor.Tensor1

class TensorIndexingTests extends FunSuite {


  test("correctly retrieves a value from a tensor") {
    val shape = Shape["Dim1", "Dim2"](2, 3)
    val t  = Tensor.fromSeq(shape, Seq(1f, 2f, 3f, 4f, 5f, 6f))
    assertEquals(t.get((0, 0)), 1f)
    assertEquals(t.get((0, 1)), 2f)
    assertEquals(t.get((0, 2)), 3f)
    assertEquals(t.get((1, 0)), 4f)
    assertEquals(t.get((1, 1)), 5f)
    assertEquals(t.get((1, 2)), 6f)
  }

   test("correclty update a value from a tensor") {
    val shape = Shape["Dim1", "Dim2"](2, 3)
    val t  = Tensor.fromSeq(shape, Seq(1f, 2f, 3f, 4f, 5f, 6f))
    t.update((1, 2), 99)
    assertEquals(t.get((0, 0)), 1f)
    assertEquals(t.get((0, 1)), 2f)
    assertEquals(t.get((0, 2)), 3f)
    assertEquals(t.get((1, 0)), 4f)
    assertEquals(t.get((1, 1)), 5f)
    assertEquals(t.get((1, 2)), 99f)
  }

}

class TensorShapeTests extends FunSuite {
  val shape = Shape["Dim1", "Dim2"](2, 3)

    test("correctly retrieves the shape of a tensor") {
        val t = Tensor.fromSeq(shape, Seq(1f, 2f, 3f, 4f, 5f, 6f))
        assertEquals(t.dim["Dim1"], 2)
        assertEquals(t.dim["Dim2"], 3)
    }
}