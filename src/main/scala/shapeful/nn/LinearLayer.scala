package shapeful.nn

import scala.language.experimental.namedTypeArguments

import shapeful.*
import shapeful.autodiff.{TensorTree, ToPyTree}
import shapeful.nn.Layer1D

class Linear[In <: Label, Out <: Label]:

  def apply(params: Linear.Params[In, Out])(input: Tensor1[In]): Tensor1[Out] =
    input.matmul(params.weight) + params.bias

object Linear:
  case class Params[InDim <: Label, OutDim <: Label](
      weight: Tensor2[InDim, OutDim],
      bias: Tensor1[OutDim]
  ) derives TensorTree,
        ToPyTree

  // Convenience method for common initialization
  def xavier[InDim <: Label, OutDim <: Label](
      key: shapeful.random.Random.Key
  )(using inDim: Dim[InDim], outDim: Dim[OutDim]): Linear.Params[InDim, OutDim] =
    val scale = math.sqrt(2.0 / (inDim.dim + outDim.dim)).toFloat
    Params(
      weight = Tensor.randn(Shape2[InDim, OutDim](inDim.dim, outDim.dim), key) * Tensor0(scale),
      bias = Tensor.zeros(Shape1[OutDim](outDim.dim))
    )

  // He initialization for ReLU activation functions
  def he[InDim <: Label, OutDim <: Label](
      key: shapeful.random.Random.Key
  )(using inDim: Dim[InDim], outDim: Dim[OutDim]): Linear.Params[InDim, OutDim] =
    val scale = math.sqrt(2.0 / inDim.dim).toFloat
    Params(
      weight = Tensor.randn(Shape2[InDim, OutDim](inDim.dim, outDim.dim), key) * Tensor0(scale),
      bias = Tensor.zeros(Shape1[OutDim](outDim.dim))
    )
