package shapeful.inference.flows

import scala.language.experimental.namedTypeArguments

import shapeful.*
import shapeful.tensor.TensorSlicing.*
import shapeful.autodiff.TensorTree
import shapeful.autodiff.ToPyTree

class AffineCouplingFlow[Dim <: Label](
    hidden_dim: Int // Hyperparameter
) extends Flow[Dim, Dim, AffineCouplingFlow.Params[Dim]]:

  def forwardSample(x: Tensor1[Dim])(params: Params): Tensor1[Dim] =
    val mask = params.mask

    // Split using mask
    val x_a = x * mask
    val x_b = x * (Tensor.ones(mask.shape) - mask)

    // Apply conditioning network using parameters
    val (log_scale, shift) = applyConditioningNetwork(x_a, params)
    // Debug the transformation parameters

    // Transform
    val scale = log_scale.exp
    val transformed_x_b = (x_b * scale) + shift

    x_a + transformed_x_b

  override def logDetJacobian(x: Tensor1[Dim])(params: Params): Tensor0 =
    val mask = params.mask
    val x_a = x * mask
    val (log_scale, _) = applyConditioningNetwork(x_a, params)

    // Log determinant = sum of log scales for transformed dimensions only
    val inverse_mask = Tensor.ones(mask.shape) - mask
    val logdet = (log_scale * inverse_mask).sum

    // Conservative clipping for stability
    logdet.clamp(-5.0f, 5.0f)

  private def applyConditioningNetwork(x: Tensor1[Dim], params: Params): (Tensor1[Dim], Tensor1[Dim]) =
    // Use the parameters from params, not internal network
    val x_input = x.relabel[Tuple1["Input"]]
    val hidden = (params.w1.transpose.matmul1(x_input) + params.b1).relu
    val output = params.w2.transpose.matmul1(hidden) + params.b2

    val output_dim = x.shape.dims(0)
    val (log_raw, shift_raw) = output.splitAt["Output"](output_dim)

    val log_scale = (log_raw.tanh * Tensor0(1.0f)).relabel[Tuple1[Dim]] // soft clamping
    val shift = (shift_raw.tanh * Tensor0(2.0f)).relabel[Tuple1[Dim]] // soft clamping

    (log_scale, shift)

object AffineCouplingFlow:

  case class Params[Dim <: Label](
      // Fixed mask - usually not optimized
      mask: Tensor1[Dim],
      // Learnable neural network parameters
      w1: Tensor2["Input", "Hidden"],
      b1: Tensor1["Hidden"],
      w2: Tensor2["Hidden", "Output"],
      b2: Tensor1["Output"]
  ) derives TensorTree,
        ToPyTree

  // Create initial parameters
  def initParams[Dim <: Label](
      input_dim: Int,
      hidden_dim: Int,
      mask: Tensor1[Dim] // Provide fixed mask
  ): Params[Dim] =
    Params(
      mask = mask,
      w1 = Tensor.randn(Shape2["Input", "Hidden"](input_dim, hidden_dim)) * Tensor0(0.01f),
      b1 = Tensor.zeros(Shape1["Hidden"](hidden_dim)),
      w2 = Tensor.randn(Shape2["Hidden", "Output"](hidden_dim, 2 * input_dim)) * Tensor0(0.01f),
      b2 = Tensor.zeros(Shape1["Output"](2 * input_dim))
    )
