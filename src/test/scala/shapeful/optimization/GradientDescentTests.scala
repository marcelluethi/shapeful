package shapeful.optimization

import munit.FunSuite
import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.tree.{TensorTree, given}
import shapeful.tree.TensorTree
import shapeful.autodiff.{Autodiff}
import shapeful.optimization.GradientOptimizer
import me.shadaj.scalapy.py

class GradientDescentTests extends FunSuite:

  // Test parameter structure
  case class SimpleScalarParam(value : Tensor0)

  // TreeMap instance for SimpleParams
  given TensorTree[SimpleScalarParam] with
    def map(p: SimpleScalarParam, f: [T <: Tuple] => Tensor[T] => Tensor[T]): SimpleScalarParam =
      SimpleScalarParam(f(p.value))

    def zipMap(tree1: SimpleScalarParam, tree2: SimpleScalarParam, f: [T <: Tuple] => (Tensor[T], Tensor[T]) => Tensor[T]): SimpleScalarParam =
      SimpleScalarParam(
        value = f(tree1.value, tree2.value)
      )
  given ToPyTree[SimpleScalarParam] with
    def toPyTree(p: SimpleScalarParam): Jax.PyAny = p.value.jaxValue
    def fromPyTree(jxpr: Jax.PyAny): SimpleScalarParam = 
        SimpleScalarParam(new Tensor(Shape0, jxpr.as[Jax.PyDynamic]))

  test("GradientOptimizer.finds minimum of simple function") {
    val optimizer = GradientOptimizer(lr = -0.1f)
    
    val initialParams = SimpleScalarParam(Tensor0(0.1f))

    // Define a simple quadratic function
    def simpleFunction(p: SimpleScalarParam): Tensor0 =
        (p.value - Tensor0(1f)).pow(Tensor0(2f))
    
    val gradient = Autodiff.grad(simpleFunction)

    // Create an iterator that applies gradient descent
    val updates = optimizer.optimize(gradient, initialParams)
    
    val finalParams = updates.take(20).toSeq.last 
    val finalValue = finalParams.value.toFloat
    
    assertEqualsDouble(finalValue, 1, 1e-1,  s"Expected value close to 1, got $finalValue")

  }
end GradientDescentTests
