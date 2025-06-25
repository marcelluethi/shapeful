package shapeful.jax

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.PyQuote

object Jax:

  type PyAny = py.Any

  // Lazy initialization to avoid errors if JAX is not installed
  lazy val sys = py.module("sys")

  // Configure Python path lazily
  private lazy val configurePythonPath: Unit =
    sys.path.append("./src")
    sys.path.append("./src/python")
    sys.path.append("../")

  // Lazy module imports with error handling
  lazy val jax =
    configurePythonPath
    try py.module("jax")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX. Make sure JAX is installed in your Python environment: ${e.getMessage}",
          e
        )

  lazy val jnp =
    configurePythonPath
    try py.module("jax.numpy")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX NumPy. Make sure JAX is installed: ${e.getMessage}",
          e
        )

  lazy val jnn =
    configurePythonPath
    try py.module("jax.nn")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import JAX Neural Network module. Make sure JAX is installed: ${e.getMessage}",
          e
        )

  lazy val jax_helper =
    configurePythonPath
    try py.module("jax_helper")
    catch
      case e: Exception =>
        throw new RuntimeException(
          s"Failed to import jax_helper module. Make sure it exists in the Python path: ${e.getMessage}",
          e
        )

trait ToPyTree[P]:
  def toPyTree(p: P): Jax.PyAny
  def fromPyTree(p: Jax.PyAny): P
