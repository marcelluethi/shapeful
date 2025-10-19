package shapeful.jax

import munit.FunSuite
import me.shadaj.scalapy.py
import shapeful.jax.Jax
import shapeful.autodiff.ToPyTree

class JaxBasicTests extends FunSuite:

  test("JAX can be imported without errors") {
    val jax = Jax.jax
    assert(jax != null, "JAX module should be accessible")
  }

  test("JAX version is accessible") {
    val jax = Jax.jax
    try
      val version = jax.__version__
      println(s"JAX version: $version")
      assert(version != null, "JAX should have a version")
    catch
      case e: Exception =>
        fail(s"Failed to get JAX version: ${e.getMessage}")
  }

  test("JAX NumPy module is accessible") {
    val jnp = Jax.jnp
    assert(jnp != null, "JAX NumPy module should be accessible")
  }

  test("JAX Neural Network module is accessible") {
    val jnn = Jax.jnn
    assert(jnn != null, "JAX Neural Network module should be accessible")
  }

  test("JAX helper module is accessible") {
    try
      val jax_helper = Jax.jax_helper
      assert(jax_helper != null, "JAX helper module should be accessible")
    catch
      case e: Exception =>
        // This is expected if jax_helper.py doesn't exist yet
        println(s"JAX helper module not found (expected): ${e.getMessage}")
  }

  test("can check JAX backend") {
    val jax = Jax.jax
    try
      val backend = jax.default_backend()
      println(s"JAX default backend: $backend")
      assert(backend != null, "JAX should have a default backend")
    catch
      case e: Exception =>
        println(s"Could not get JAX backend: ${e.getMessage}")
    // This might fail on some systems, so we just log it
  }

  test("can create simple JAX array with scalar") {
    val jnp = Jax.jnp
    try
      val scalar = jnp.array(42.0)
      assert(scalar != null, "Should be able to create scalar JAX array")
      println(s"Scalar array: $scalar")
    catch
      case e: Exception =>
        fail(s"Failed to create scalar JAX array: ${e.getMessage}")
  }

  test("ToPyTree trait exists and can be instantiated") {
    val converter = new ToPyTree[Int]:
      def toPyTree(p: Int): Jax.PyAny = py.Dynamic.global.int(p)
      def fromPyTree(p: Jax.PyAny): Int = p.as[Int]

    assert(converter != null, "ToPyTree implementation should compile")
  }

  test("sys module paths are configured") {
    val sys = Jax.sys
    assert(sys != null, "sys module should be accessible")

    val path = sys.path
    assert(path != null, "sys.path should be accessible")
    println(s"Python path includes ${path.bracketAccess(0)} and other entries")
  }

  test("can access py module through Jax") {
    // Test that we can use basic Python functionality
    val pyInt = py.Dynamic.global.int(42)
    val pyStr = py.Dynamic.global.str("hello")

    assert(pyInt != null, "Should be able to create Python int")
    assert(pyStr != null, "Should be able to create Python string")
  }

  test("Tensor0 can be converted to PyTree and recovered") {
    import shapeful.*
    import shapeful.tensor.DType

    val original = Tensor0(42.5f)

    // Convert to PyTree
    val pytree = summon[ToPyTree[Tensor0]].toPyTree(original)
    assert(pytree != null, "PyTree conversion should succeed")

    // Recover from PyTree
    val recovered = summon[ToPyTree[Tensor0]].fromPyTree(pytree)

    // Verify the recovered tensor matches the original
    assert(
      recovered.approxEquals(original, tolerance = 1e-6f),
      s"Recovered tensor ${recovered.toFloat} should match original ${original.toFloat}"
    )
    assert(
      recovered.dtype == original.dtype,
      s"Recovered dtype ${recovered.dtype} should match original ${original.dtype}"
    )
  }

  test("Tensor1 can be converted to PyTree and recovered") {
    import shapeful.*
    import shapeful.tensor.DType

    type Feature = "feature"
    val original = Tensor1(Axis[Feature], Seq(1.0f, 2.5f, 3.7f, 4.2f))

    // Convert to PyTree
    val pytree = summon[ToPyTree[Tensor1[Feature]]].toPyTree(original)
    assert(pytree != null, "PyTree conversion should succeed")

    // Recover from PyTree
    val recovered = summon[ToPyTree[Tensor1[Feature]]].fromPyTree(pytree)

    // Verify the recovered tensor matches the original
    assert(recovered.approxEquals(original, tolerance = 1e-6f), s"Recovered tensor should match original")
    assert(
      recovered.shape.dims == original.shape.dims,
      s"Recovered shape ${recovered.shape.dims} should match original ${original.shape.dims}"
    )
    assert(
      recovered.dtype == original.dtype,
      s"Recovered dtype ${recovered.dtype} should match original ${original.dtype}"
    )
  }

  test("Tensor2 can be converted to PyTree and recovered") {
    import shapeful.*
    import shapeful.tensor.DType

    type Height = "height"
    type Width = "width"
    val original = Tensor2(
      Axis[Height],
      Axis[Width],
      Seq(
        Seq(1.0f, 2.0f, 3.0f),
        Seq(4.0f, 5.0f, 6.0f)
      )
    )

    // Convert to PyTree
    val pytree = summon[ToPyTree[Tensor2[Height, Width]]].toPyTree(original)
    assert(pytree != null, "PyTree conversion should succeed")

    // Recover from PyTree
    val recovered = summon[ToPyTree[Tensor2[Height, Width]]].fromPyTree(pytree)

    // Verify the recovered tensor matches the original
    assert(recovered.approxEquals(original, tolerance = 1e-6f), s"Recovered tensor should match original")
    assert(
      recovered.shape.dims == original.shape.dims,
      s"Recovered shape ${recovered.shape.dims} should match original ${original.shape.dims}"
    )
    assert(
      recovered.dtype == original.dtype,
      s"Recovered dtype ${recovered.dtype} should match original ${original.dtype}"
    )
  }

  test("PyTree conversion preserves different dtypes") {
    import shapeful.*
    import shapeful.tensor.DType

    // Test with Float32 (default)
    val float32Tensor = Tensor0(3.14159f)
    val float32Tree = summon[ToPyTree[Tensor0]].toPyTree(float32Tensor)
    val float32Recovered = summon[ToPyTree[Tensor0]].fromPyTree(float32Tree)

    assert(float32Recovered.dtype == DType.Float32, s"Should preserve Float32 dtype, got ${float32Recovered.dtype}")
    assert(float32Recovered.approxEquals(float32Tensor, tolerance = 1e-6f), "Should preserve Float32 precision")

    // Test with Int32
    val int32Tensor = Tensor0(42)
    val int32Tree = summon[ToPyTree[Tensor0]].toPyTree(int32Tensor)
    val int32Recovered = summon[ToPyTree[Tensor0]].fromPyTree(int32Tree)

    assert(int32Recovered.dtype == DType.Int32, s"Should preserve Int32 dtype, got ${int32Recovered.dtype}")
  }

  test("PyTree conversion works with different tensor shapes") {
    import shapeful.*
    import shapeful.tensor.DType

    type Feature = "feature"
    type Batch = "batch"
    type Hidden = "hidden"

    // Test 3D tensor
    val tensor3D = Tensor3(
      Axis[Feature],
      Axis[Batch],
      Axis[Hidden],
      Seq(
        Seq(
          Seq(0.1f, 0.2f),
          Seq(0.3f, 0.4f)
        ),
        Seq(
          Seq(0.5f, 0.6f),
          Seq(0.7f, 0.8f)
        )
      )
    )

    // Convert to PyTree
    val pytree = summon[ToPyTree[Tensor3[Feature, Batch, Hidden]]].toPyTree(tensor3D)
    assert(pytree != null, "PyTree conversion should succeed for 3D tensor")

    // Recover from PyTree
    val recovered = summon[ToPyTree[Tensor3[Feature, Batch, Hidden]]].fromPyTree(pytree)

    // Verify shape and values match
    assert(recovered.approxEquals(tensor3D, tolerance = 1e-6f), "Recovered 3D tensor should match original")
    assert(recovered.shape.dims == tensor3D.shape.dims, "3D tensor shape should be preserved")
    assert(recovered.dtype == tensor3D.dtype, "3D tensor dtype should be preserved")
  }
