package shapeful.tensor

import munit.FunSuite
import shapeful.tensor.Tensor.Tensor0
import shapeful.tensor.Tensor
import shapeful.tensor.TensorOps.* 
import shapeful.tensor.Tensor0Ops.*
import shapeful.tensor.Tensor.Tensor1

class ShapeTests extends FunSuite {


  test("correctly builds cartesian product of two shapes") {
    val shape = Shape["Dim1", "Dim2"](2, 3)
    val shape2 = Shape["Dim3", "Dim4"](4, 5)
    val newShape = shape *: shape2

    assertEquals(newShape.dims, List(2, 3, 4, 5))
  }

  test("Correctly handles update operations") {}

  test("Correctly handles update eoperation for duplicate keys") {}

  test("Correctly handles update operation for non-existing keys") {}

  test("Correctly handles remove operation") {}

  test("Correctly handles remove operation for non-existing keys") {}

  test("Correctly handles remove operation for duplicate keys") {}
}