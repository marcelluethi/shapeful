package shapeful.nn

import scala.language.experimental.namedTypeArguments

import shapeful.*
import shapeful.autodiff.{TensorTree, ToPyTree}
import shapeful.nn.Layer1D

class Linear[In <: Label, Out <: Label]() extends Layer1D[In, Out, Linear.Params[In, Out]]:

  override def forward(params: Linear.Params[In, Out]): Function1[Tensor1[In], Tensor1[Out]] =
    input =>
      val weight = params.weight
      val bias = params.bias
      input.matmul(weight) + bias

object Linear:
  case class Params[InDim <: Label, OutDim <: Label](
      weight: Tensor2[InDim, OutDim],
      bias: Tensor1[OutDim]
  ) derives TensorTree,
        ToPyTree

  // Convenience method for common initialization
  def xavier[InDim <: Label, OutDim <: Label](
      input_dim: Int,
      output_dim: Int
  ): Linear.Params[InDim, OutDim] =
    val scale = math.sqrt(2.0 / (input_dim + output_dim)).toFloat
    Params(
      weight = Tensor.randn(Shape2[InDim, OutDim](input_dim, output_dim)) * Tensor0(scale),
      bias = Tensor.zeros(Shape1[OutDim](output_dim))
    )

  // He initialization for ReLU activation functions
  def he[InDim <: Label, OutDim <: Label](
      input_dim: Int,
      output_dim: Int
  ): Linear.Params[InDim, OutDim] =
    val scale = math.sqrt(2.0 / input_dim).toFloat
    Params(
      weight = Tensor.randn(Shape2[InDim, OutDim](input_dim, output_dim)) * Tensor0(scale),
      bias = Tensor.zeros(Shape1[OutDim](output_dim))
    )
