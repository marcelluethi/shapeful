package shapeful.nn

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import scala.annotation.targetName

object Utils:

  def oneHot[Input <: Label, Classes <: Label](
      labels: Tensor1[Input],
      numClasses: Int
  ): Tensor2[Input, Classes] =
    val result = Jax.jnn.one_hot(labels.jaxValue, numClasses)
    new Tensor(Shape2[Input, Classes](labels.shape.dim[Input], numClasses), result, labels.dtype)

  @targetName("oneHotScalar")
  def oneHot[Classes <: Label](
      labels: Tensor0,
      numClasses: Int
  ): Tensor1[Classes] =
    val result = Jax.jnn.one_hot(labels.jaxValue, numClasses)
    new Tensor(Shape1[Classes](numClasses), result, labels.dtype)
