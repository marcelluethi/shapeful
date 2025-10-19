package shapeful.tensor

import munit.FunSuite
import shapeful.*
import shapeful.tensor.Shape.*

class TensorContractTests extends FunSuite:

  // Define labels for testing
  type Batch = "batch"
  type Rows = "rows"
  type Cols = "cols"
  type Features = "features"
  type Inner = "inner"
  type Dim = "dim"
  type I = "i"
  type J = "j"
  type K = "k"

  override def beforeAll(): Unit =
    super.beforeAll()

  test("contract: matrix-vector multiplication") {
    // Matrix: (Rows, Features) = [[1, 2], [3, 4]]
    val matrix = Tensor2(Axis[Rows], Axis[Features], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))

    // Vector: (Features,) = [5, 6]
    val vector = Tensor1(Axis[Features], Seq(5.0f, 6.0f))

    // Contract over Features dimension
    val result: Tensor1[Rows] = matrix.contract(Axis[Features], vector)

    // Expected: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    assertEquals(result.shape.size, 2, "size")

    val expected = Tensor1(Axis[Rows], Seq(17.0f, 39.0f))
    assert(
      result.approxEquals(expected, tolerance = 1e-5f),
      s"Matrix-vector contraction failed.\nExpected: ${expected}\nActual: ${result}"
    )
  }

  test("contract: matrix-matrix multiplication") {
    // m1: (Rows, Inner) = [[1, 2], [3, 4]]
    val m1 = Tensor2(Axis[Rows], Axis[Inner], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))

    // m2: (Inner, Cols) = [[5, 6], [7, 8]]
    val m2 = Tensor2(Axis[Inner], Axis[Cols], Seq(Seq(5.0f, 6.0f), Seq(7.0f, 8.0f)))

    // Contract over Inner dimension
    val result: Tensor2[Rows, Cols] = m1.contract(Axis[Inner], m2)

    // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    //         = [[19, 22], [43, 50]]
    assertEquals(result.shape.dims, Seq(2, 2), "dims")

    val expected = Tensor2(Axis[Rows], Axis[Cols], Seq(Seq(19.0f, 22.0f), Seq(43.0f, 50.0f)))
    assert(
      result.approxEquals(expected, tolerance = 1e-5f),
      s"Matrix-matrix contraction failed.\nExpected: ${expected}\nActual: ${result}"
    )
  }

  test("contract: batched operations") {
    // batched: (Batch, Rows, Inner) with shape (2, 2, 2)
    val batched = Tensor3(
      Axis[Batch],
      Axis[Rows],
      Axis[Inner],
      Seq(
        Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)), // First batch
        Seq(Seq(5.0f, 6.0f), Seq(7.0f, 8.0f)) // Second batch
      )
    )

    // weights: (Inner, Cols) = [[1, 2], [3, 4]]
    val weights = Tensor2(Axis[Inner], Axis[Cols], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))

    // Contract over Inner, keeping Batch and Rows
    val result: Tensor3[Batch, Rows, Cols] = batched.contract(Axis[Inner], weights)

    assertEquals(result.shape.dims, Seq(2, 2, 2), "dims")

    // Expected for first batch: [[1*1 + 2*3, 1*2 + 2*4], [3*1 + 4*3, 3*2 + 4*4]]
    //                          = [[7, 10], [15, 22]]
    // Expected for second batch: [[5*1 + 6*3, 5*2 + 6*4], [7*1 + 8*3, 7*2 + 8*4]]
    //                           = [[23, 34], [31, 46]]
    val expected = Tensor3(
      Axis[Batch],
      Axis[Rows],
      Axis[Cols],
      Seq(
        Seq(Seq(7.0f, 10.0f), Seq(15.0f, 22.0f)),
        Seq(Seq(23.0f, 34.0f), Seq(31.0f, 46.0f))
      )
    )
    assert(
      result.approxEquals(expected, tolerance = 1e-5f),
      s"Batched contraction failed.\nExpected: ${expected}\nActual: ${result}"
    )
  }

  test("contract: dot product (scalar result)") {
    // v1: (Dim,) = [1, 2, 3]
    val v1 = Tensor1(Axis[Dim], Seq(1.0f, 2.0f, 3.0f))

    // v2: (Dim,) = [4, 5, 6]
    val v2 = Tensor1(Axis[Dim], Seq(4.0f, 5.0f, 6.0f))

    // Contract over Dim, returns scalar
    val result: Tensor0 = v1.contract(Axis[Dim], v2)

    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assertEquals(result.shape.dims, Seq.empty, "dims")
    assertEqualsDouble(result.toFloat.toDouble, 32.0, 1e-5)
  }

  test("contract: vector-matrix multiplication") {
    // vector: (Inner,) = [1, 2, 3]
    val vector = Tensor1(Axis[Inner], Seq(1.0f, 2.0f, 3.0f))

    // matrix: (Inner, Cols) = [[1, 2], [3, 4], [5, 6]]
    val matrix = Tensor2(Axis[Inner], Axis[Cols], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f), Seq(5.0f, 6.0f)))

    // Contract over Inner dimension
    val result: Tensor1[Cols] = vector.contract(Axis[Inner], matrix)

    // Expected: [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    assertEquals(result.shape.size, 2, "size")

    val expected = Tensor1(Axis[Cols], Seq(22.0f, 28.0f))
    assert(
      result.approxEquals(expected, tolerance = 1e-5f),
      s"Vector-matrix contraction failed.\nExpected: ${expected}\nActual: ${result}"
    )
  }

  test("contract: outer product becomes inner product when contracted") {
    // v1: (I,) = [1, 2]
    val v1 = Tensor1(Axis[I], Seq(1.0f, 2.0f))

    // v2: (I,) = [3, 4]
    val v2 = Tensor1(Axis[I], Seq(3.0f, 4.0f))

    // Contract over I dimension
    val result: Tensor0 = v1.contract(Axis[I], v2)

    // Expected: 1*3 + 2*4 = 11
    assertEqualsDouble(result.toFloat.toDouble, 11.0, 1e-5)
  }

  test("contract: 3D tensor with 2D tensor") {
    // t1: (Batch, I, J) with shape (2, 2, 3)
    val t1 = Tensor3(
      Axis[Batch],
      Axis[I],
      Axis[J],
      Seq(
        Seq(Seq(1.0f, 2.0f, 3.0f), Seq(4.0f, 5.0f, 6.0f)),
        Seq(Seq(7.0f, 8.0f, 9.0f), Seq(10.0f, 11.0f, 12.0f))
      )
    )

    // t2: (J, K) with shape (3, 2)
    val t2 = Tensor2(Axis[J], Axis[K], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f), Seq(5.0f, 6.0f)))

    // Contract over J dimension
    val result: Tensor3[Batch, I, K] = t1.contract(Axis[J], t2)

    assertEquals(result.shape.dims, Seq(2, 2, 2), "dims")

    // Verify a few values manually
    // For batch 0, row 0: [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    // For batch 0, row 1: [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [49, 64]
    val expected = Tensor3(
      Axis[Batch],
      Axis[I],
      Axis[K],
      Seq(
        Seq(Seq(22.0f, 28.0f), Seq(49.0f, 64.0f)),
        Seq(Seq(76.0f, 100.0f), Seq(103.0f, 136.0f))
      )
    )
    assert(
      result.approxEquals(expected, tolerance = 1e-5f),
      s"3D-2D contraction failed.\nExpected: ${expected}\nActual: ${result}"
    )
  }

  test("contract: consistency with existing matmul for 2D tensors") {
    // Compare contract with existing matmul operation
    val m1 = Tensor2(Axis[Rows], Axis[Inner], Seq(Seq(1.0f, 2.0f, 3.0f), Seq(4.0f, 5.0f, 6.0f)))

    val m2 = Tensor2(Axis[Inner], Axis[Cols], Seq(Seq(7.0f, 8.0f), Seq(9.0f, 10.0f), Seq(11.0f, 12.0f)))

    // Using contract
    val contractResult: Tensor2[Rows, Cols] = m1.contract(Axis[Inner], m2)

    // Using existing matmul
    val matmulResult = m1.matmul(m2)

    assert(
      contractResult.approxEquals(matmulResult, tolerance = 1e-5f),
      s"Contract should match matmul for 2D tensors.\nContract: ${contractResult}\nMatmul: ${matmulResult}"
    )
  }

  test("contract: consistency with dot for 1D tensors") {
    // Compare contract with existing dot operation
    val v1 = Tensor1(Axis[Dim], Seq(1.0f, 2.0f, 3.0f, 4.0f))
    val v2 = Tensor1(Axis[Dim], Seq(5.0f, 6.0f, 7.0f, 8.0f))

    // Using contract
    val contractResult: Tensor0 = v1.contract(Axis[Dim], v2)

    // Using existing dot
    val dotResult = v1.dot(v2)

    assert(
      contractResult.approxEquals(dotResult, tolerance = 1e-5f),
      s"Contract should match dot for 1D tensors.\nContract: ${contractResult}\nDot: ${dotResult}"
    )
  }

  test("contract: different batch sizes retained in result") {
    // Test that non-contracted dimensions keep their sizes correctly
    val t1 = Tensor2(
      Axis[Batch],
      Axis[Inner],
      Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f), Seq(5.0f, 6.0f)) // 3 batches
    )

    val t2 = Tensor2(
      Axis[Inner],
      Axis[Features],
      Seq(Seq(1.0f, 2.0f, 3.0f), Seq(4.0f, 5.0f, 6.0f)) // 2 inner, 3 features
    )

    val result: Tensor2[Batch, Features] = t1.contract(Axis[Inner], t2)

    // Result should have shape (3, 3) - 3 batches, 3 features
    assertEquals(result.shape.dims, Seq(3, 3), "dims")

    // Verify first batch: [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
    val expected = Tensor2(
      Axis[Batch],
      Axis[Features],
      Seq(
        Seq(9.0f, 12.0f, 15.0f),
        Seq(19.0f, 26.0f, 33.0f),
        Seq(29.0f, 40.0f, 51.0f)
      )
    )
    assert(
      result.approxEquals(expected, tolerance = 1e-5f),
      s"Batch dimension sizes incorrect.\nExpected: ${expected}\nActual: ${result}"
    )
  }
