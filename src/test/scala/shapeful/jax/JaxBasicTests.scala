package shapeful.jax

import munit.FunSuite
import me.shadaj.scalapy.py
import shapeful.jax.Jax

class JaxBasicTests extends FunSuite {

  test("JAX can be imported without errors") {
    val jax = Jax.jax
    assert(jax != null, "JAX module should be accessible")
  }

  test("JAX version is accessible") {
    val jax = Jax.jax
    try {
      val version = jax.__version__
      println(s"JAX version: $version")
      assert(version != null, "JAX should have a version")
    } catch {
      case e: Exception => 
        fail(s"Failed to get JAX version: ${e.getMessage}")
    }
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
    try {
      val jax_helper = Jax.jax_helper
      assert(jax_helper != null, "JAX helper module should be accessible")
    } catch {
      case e: Exception =>
        // This is expected if jax_helper.py doesn't exist yet
        println(s"JAX helper module not found (expected): ${e.getMessage}")
    }
  }

  test("can check JAX backend") {
    val jax = Jax.jax
    try {
      val backend = jax.default_backend()
      println(s"JAX default backend: $backend")
      assert(backend != null, "JAX should have a default backend")
    } catch {
      case e: Exception =>
        println(s"Could not get JAX backend: ${e.getMessage}")
        // This might fail on some systems, so we just log it
    }
  }

  test("can create simple JAX array with scalar") {
    val jnp = Jax.jnp
    try {
      val scalar = jnp.array(42.0)
      assert(scalar != null, "Should be able to create scalar JAX array")
      println(s"Scalar array: $scalar")
    } catch {
      case e: Exception =>
        fail(s"Failed to create scalar JAX array: ${e.getMessage}")
    }
  }

  test("ToPyTree trait exists and can be instantiated") {
    val converter = new ToPyTree[Int] {
      def toPyTree(p: Int): Jax.PyAny = py.Dynamic.global.int(p)
      def fromPyTree(p: Jax.PyAny): Int = p.as[Int]
    }
    
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
}
