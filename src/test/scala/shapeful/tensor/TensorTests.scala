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
    assertEquals(t.select(Shape["Dim1", "Dim2"](0, 0)).item, 1f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](0, 1)).item, 2f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](0, 2)).item, 3f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](1, 0)).item, 4f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](1, 1)).item, 5f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](1, 2)).item, 6f)
  }

   test("correclty update a value from a tensor") {
    val shape = Shape["Dim1", "Dim2"](2, 3)
    val t  = Tensor.fromSeq(shape, Seq(1f, 2f, 3f, 4f, 5f, 6f))
    t.update((1, 2), 99)
    assertEquals(t.select(Shape["Dim1", "Dim2"](0, 0)).item, 1f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](0, 1)).item, 2f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](0, 2)).item, 3f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](1, 0)).item, 4f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](1, 1)).item, 5f)
    assertEquals(t.select(Shape["Dim1", "Dim2"](1, 2)).item, 99f)
  }

}

class TensorShapeTests extends FunSuite {
  val shape = Shape["Dim1", "Dim2"](2, 4)
  val t = Tensor.fromSeq(shape, Seq(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))

    test("correctly retrieves the shape of a tensor") {

        assertEquals(t.dim["Dim1"], 2)
        assertEquals(t.dim["Dim2"], 4)
    }

    test("correctly reshapes a tensor") {
        val newShape = Shape["Dim1", "Dim2", "Dim3"](2, 2, 2)
        val reshaped = t.reshape(newShape)
        assertEquals(reshaped.dim["Dim1"], 2)
        assertEquals(reshaped.dim["Dim2"], 2)
        assertEquals(reshaped.dim["Dim3"], 2)
        for {
          i <- 0 until 2
          j <- 0 until 2
          k <- 0 until 2
        } {
          assertEquals(reshaped.select(Shape["Dim1", "Dim2", "Dim3"](i, j, k)).item.toInt, i * 4 + j * 2 + k + 1)
        }
    }
}
