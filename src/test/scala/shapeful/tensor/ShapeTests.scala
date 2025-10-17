package shapeful.tensor

import munit.FunSuite
import shapeful.tensor.Shape.*

class ShapeTests extends FunSuite:

  test("dim method works with labels") {
    val shape = Shape(Axis["height"] -> 50, Axis["width"] -> 100)

    assertEquals(shape.dim["height"], 50)
    assertEquals(shape.dim["width"], 100)
  }

  test("Shape concatenation with *: operator works") {
    val shape1 = Shape(Axis["a"] -> 3)
    val shape2 = Shape(Axis["b"] -> 4, Axis["c"] -> 5)

    val concatenated = shape1 *: shape2
    assertEquals(concatenated.dims, Seq(3, 4, 5))
    assertEquals(concatenated.rank, 3)
    assertEquals(concatenated.size, 60)
  }

  test("Shape0 is empty") {
    assertEquals(Shape0.dims, Seq.empty)
    assertEquals(Shape0.rank, 0)
    assertEquals(Shape0.size, 1) // Empty product is 1
  }

  test("concatenation with empty shape works") {
    val shape = Shape(Axis["a"] -> 3, Axis["b"] -> 4)
    val withEmpty1 = Shape0 *: shape
    val withEmpty2 = shape *: Shape0

    assertEquals(withEmpty1.dims, Seq(3, 4))
    assertEquals(withEmpty2.dims, Seq(3, 4))
  }

  test("relabel preserves dimensions") {
    val original = Shape(Axis["old1"] -> 10, Axis["old2"] -> 20)
    val relabeled = original.relabel["new1" *: "new2" *: EmptyTuple]

    assertEquals(relabeled.dims, Seq(10, 20))
    assertEquals(relabeled.rank, 2)
    assertEquals(relabeled.size, 200)
  }

  test("asTuple conversion works correctly") {
    val shape1 = Shape(Axis["a"] -> 5)
    val shape2 = Shape(Axis["x"] -> 3, Axis["y"] -> 4)
    val shape3 = Shape(Axis["i"] -> 2, Axis["j"] -> 3, Axis["k"] -> 4)

    assertEquals(shape1.asTuple, Tuple1(5))
    assertEquals(shape2.asTuple, (3, 4))
    assertEquals(shape3.asTuple, (2, 3, 4))
  }
