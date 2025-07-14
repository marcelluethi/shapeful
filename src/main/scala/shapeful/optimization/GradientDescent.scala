package shapeful.optimization

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.tensor.{Tensor, TensorOps}
import shapeful.tree.TensorTree

class GradientDescent(lr: Float):

  /** Creates an iterator that applies gradient descent optimization. Each iteration computes gradients and updates
    * parameters using TreeMap.
    *
    * This implementation requires that Params and Gradient have the same type for proper gradient application (which is
    * the common case).
    *
    * @param df
    *   Function that computes gradients given parameters
    * @param initialParams
    *   Starting parameter values
    * @return
    *   Iterator over parameter updates
    */
  def optimize[Params](
      df: Params => Params,
      initialParams: Params
  )(using paramTree: TensorTree[Params]) =
    Iterator.iterate(initialParams) { params =>
      val gradients = df(params)

      paramTree.zipMap(gradients, params, [T <: Tuple] => (g, p) => p - (g * Tensor0(lr)))
    }
