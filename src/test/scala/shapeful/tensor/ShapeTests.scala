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
    val newShape = shape ++ shape2

    assertEquals(newShape.dims, List(2, 3, 4, 5))
  }

  test("Correctly handles update operations") {
    val shape = Shape["Dim1", "Dim2", "Dim3", "Dim4"](1, 2, 3, 4)
    assertEquals(shape.updateValue["Dim1"](11).dims, List(11, 2, 3, 4))
    assertEquals(shape.updateValue["Dim2"](22).dims, List(1, 22, 3, 4))
    assertEquals(shape.updateValue["Dim3"](33).dims, List(1, 2, 33, 4))    
    assertEquals(shape.updateValue["Dim4"](44).dims, List(1, 2, 3, 44))    
  }

  test("Correctly handles remove operation") {
    val shape = Shape["Dim1", "Dim2", "Dim3", "Dim4"](1, 2, 3, 4)
    assertEquals(shape.removeKey["Dim1"].dims, List(2, 3, 4))
    assertEquals(shape.removeKey["Dim2"].dims, List(1, 3, 4))
    assertEquals(shape.removeKey["Dim3"].dims, List(1, 2, 4))
    assertEquals(shape.removeKey["Dim4"].dims, List(1, 2, 3))
  }

  test("Correctly handles rename operation") {
    val shape = Shape["Dim1", "Dim2", "Dim3", "Dim4"](1, 2, 3, 4)
    val newShape = shape.rename[("Dim5", "Dim6", "Dim7", "Dim8")]
    assertEquals(newShape.dim["Dim5"], 1)
    assertEquals(newShape.dim["Dim6"], 2)
    assertEquals(newShape.dim["Dim7"], 3)
    assertEquals(newShape.dim["Dim8"], 4)
  }
    

  test("Correctly gives int index of multiple dimensions") {
    val shape = Shape["Dim1", "Dim2", "Dim3", "Dim4"](1, 2, 3, 4)
    assertEquals(shape.dimsWithIndex[("Dim3", "Dim1", "Dim4")], Seq((3, 2), (1, 0), (4, 3)))    
  }

}