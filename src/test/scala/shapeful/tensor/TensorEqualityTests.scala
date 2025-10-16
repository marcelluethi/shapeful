package shapeful.tensor

import munit.FunSuite

class TensorEqualityTests extends FunSuite:

  test("basic tensor equality") {
    val shape = Shape2["height", "width"](2, 2)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)

    // Test our custom equality method
    assert(tensor1.tensorEquals(tensor2))

    // Test operator overloads
    assert(tensor1 == tensor2)
    assert(!(tensor1 != tensor2))
  }

  test("tensor inequality") {
    val shape = Shape2["height", "width"](2, 2)
    val values1 = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val values2 = Seq(1.0f, 2.0f, 3.0f, 5.0f) // Last value different

    val tensor1 = Tensor(shape, values1)
    val tensor2 = Tensor(shape, values2)

    assert(!tensor1.tensorEquals(tensor2))
    assert(tensor1 != tensor2)
    assert(!(tensor1 == tensor2))
  }

  test("scalar tensor equality") {
    val tensor1 = Tensor0(42.0f)
    val tensor2 = Tensor0(42.0f)
    val tensor3 = Tensor0(43.0f)

    assert(tensor1.tensorEquals(tensor2))
    assert(!tensor1.tensorEquals(tensor3))
  }

end TensorEqualityTests
