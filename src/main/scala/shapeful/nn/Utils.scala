package shapeful.nn

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax

object Utils:

  def oneHot[Input <: Label, Classes <: Label](
      labels: Tensor1[Input],
      numClasses: Int
  ): Tensor2[Input, Classes] =
    val result = Jax.jnn.one_hot(labels.jaxValue, numClasses)
    new Tensor(Shape2[Input, Classes](labels.shape.dim[Input], numClasses), result, labels.dtype)
