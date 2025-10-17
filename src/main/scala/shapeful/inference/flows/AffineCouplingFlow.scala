package shapeful.inference.flows

import shapeful.*
import shapeful.tensor.TensorSlicing.*
import shapeful.autodiff.{TensorTree, ToPyTree}
import shapeful.nn.{Layer1D, Linear}
import shapeful.tensor.TensorOps
import shapeful.nn.Activation
import shapeful.inference.flows.AffineCouplingFlow.NetworkParams
import shapeful.jax.Jax

class AffineCouplingFlow[Dim <: Label](mask: Tensor1[Dim]) extends Flow[Dim, Dim, AffineCouplingFlow.Params[Dim]]:

  // Two separate networks for clarity
  private lazy val scaleNetwork = new Layer1D["Input", "Scale", NetworkParams]:
    val hiddenLayer = new Linear["Input", "Hidden"]()
    val logScaleLayer = new Linear["Hidden", "Scale"]()

    def forward(params: NetworkParams): Tensor1["Input"] => Tensor1["Scale"] =
      hiddenLayer(params.logScaleHiddenLayer).andThen(logScaleLayer(params.logScaleOutputLayer))

  private lazy val shiftNetwork = new Layer1D["Input", "Shift", NetworkParams]:
    val hiddenLayer = new Linear["Input", "Hidden"]()
    val shiftLayer = new Linear["Hidden", "Shift"]()

    def forward(params: NetworkParams): Tensor1["Input"] => Tensor1["Shift"] =
      hiddenLayer(params.shiftHiddenLayer).andThen(Activation.relu).andThen(shiftLayer(params.shiftOutputLayer))

  def forwardSample(x: Tensor1[Dim])(params: Params): Tensor1[Dim] =
    val input_dim = mask.shape.dim[Dim] // Infer input_dim from mask
    val inverse_mask = Tensor.ones(mask.shape) - mask
    // Split using mask
    val x_a = x * mask // These dimensions stay unchanged
    val x_b = x * inverse_mask // These dimensions get transformed

    // Apply conditioning network using unchanged dimensions (x_a)
    val (logScale, shift) = applyConditioningNetwork(x_a, params, input_dim)
    val scale = logScale.exp // Ensure scale is positive

    // Apply transformation only to the dimensions that should be transformed
    val transformed_x_b = (x_b * (scale * inverse_mask)) + (shift * inverse_mask)

    x_a + transformed_x_b

  private def applyConditioningNetwork(x: Tensor1[Dim], params: Params, input_dim: Int): (Tensor1[Dim], Tensor1[Dim]) =
    val x_input = x.relabel[Tuple1["Input"]]

    // Networks now output full dimension
    val scale = scaleNetwork.forward(params.networkParams)(x_input).relabel[Tuple1[Dim]] // Shape: [input_dim]
    val shift = shiftNetwork.forward(params.networkParams)(x_input).relabel[Tuple1[Dim]] // Shape: [input_dim]

    (scale, shift)

object AffineCouplingFlow:

  // Network parameters for the conditioning network
  case class NetworkParams(
      logScaleHiddenLayer: Linear.Params["Input", "Hidden"],
      logScaleOutputLayer: Linear.Params["Hidden", "Scale"],
      shiftHiddenLayer: Linear.Params["Input", "Hidden"],
      shiftOutputLayer: Linear.Params["Hidden", "Shift"]
  ) derives TensorTree,
        ToPyTree

  // Main flow parameters
  case class Params[Dim <: Label](
      // Conditioning network parameters
      networkParams: NetworkParams
  ) derives TensorTree,
        ToPyTree

  def initParams[Dim <: Label](
      mask: Tensor1[Dim],
      hidden_dim: Int,
      key: shapeful.random.Random.Key
  ): Params[Dim] =
    println("createing flow with mask: " + mask)

    val input_dim = mask.shape.dim[Dim]
    val output_dim = input_dim // Full dimension output

    val keys = key.split(4)

    val networkParams = NetworkParams(
      logScaleHiddenLayer = Linear.Params(
        weight =
          Tensor.randn(keys(0), Shape(Axis["Input"] -> input_dim, Axis["Hidden"] -> hidden_dim)) * Tensor0(0.01f),
        bias = Tensor.zeros(Shape(Axis["Hidden"] -> hidden_dim))
      ),
      logScaleOutputLayer = Linear.Params(
        weight =
          Tensor.randn(keys(1), Shape(Axis["Hidden"] -> hidden_dim, Axis["Scale"] -> output_dim)) * Tensor0(0.01f),
        bias = Tensor.zeros(Shape(Axis["Scale"] -> output_dim))
      ),
      shiftHiddenLayer = Linear.Params(
        weight = Tensor.randn(keys(2), Shape(Axis["Input"] -> input_dim, Axis["Hidden"] -> hidden_dim)) * Tensor0(1f),
        bias = Tensor.zeros(Shape(Axis["Hidden"] -> hidden_dim))
      ),
      shiftOutputLayer = Linear.Params(
        weight = Tensor.randn(keys(3), Shape(Axis["Hidden"] -> hidden_dim, Axis["Shift"] -> output_dim)) * Tensor0(1f),
        bias = Tensor.zeros(Shape(Axis["Shift"] -> output_dim))
      )
    )
    Params(networkParams = networkParams)
