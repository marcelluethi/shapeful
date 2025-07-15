package examples.advanced

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.autodiff.*

/**
 * Advanced PyTree customization examples.
 * 
 * This example demonstrates:
 * - Creating custom PyTree instances for complex data structures
 * - Working with nested structures
 * - Using PyTree instances with autodiff
 * - Best practices for PyTree design
 */
object CustomPyTree extends App:

  println("=== Custom PyTree Examples ===\n")

  type Feature = "feature"
  type Hidden = "hidden"
  type Output = "output"
  type Layer = "layer"

  // 1. Simple custom structure
  println("1. Simple Custom PyTree Structure")
  
  case class SimpleModel(
    weight: Tensor2[Feature, Hidden],
    bias: Tensor1[Hidden],
    scale: Tensor0
  )
  
  // The ToPyTree instance is automatically available for tuples,
  // but let's see how it works with our simple structure
  
  val simpleModel = SimpleModel(
    weight = Tensor2[Feature, Hidden](Seq(
      Seq(0.1f, 0.2f, 0.3f),
      Seq(0.4f, 0.5f, 0.6f)
    )),
    bias = Tensor1[Hidden](Seq(0.1f, 0.0f, -0.1f)),
    scale = Tensor0(1.0f)
  )
  
  println(s"Simple model created:")
  println(s"  weight shape: ${simpleModel.weight.shape.dims}")
  println(s"  bias shape: ${simpleModel.bias.shape.dims}")
  println(s"  scale: ${simpleModel.scale}")
  println()

  // 2. Nested structure example
  println("2. Nested PyTree Structure")
  
  case class NetworkLayer(
    weight: Tensor2[Feature, Hidden],
    bias: Tensor1[Hidden]
  )
  
  case class MultiLayerModel(
    layer1: NetworkLayer,
    layer2: NetworkLayer,
    outputWeight: Tensor2[Hidden, Output],
    outputBias: Tensor1[Output],
    globalScale: Tensor0
  )
  
  val multiLayerModel = MultiLayerModel(
    layer1 = NetworkLayer(
      weight = Tensor2[Feature, Hidden](Seq(
        Seq(0.1f, 0.2f),
        Seq(0.3f, 0.4f),
        Seq(0.5f, 0.6f)
      )),
      bias = Tensor1[Hidden](Seq(0.1f, 0.0f))
    ),
    layer2 = NetworkLayer(
      weight = Tensor2[Feature, Hidden](Seq(  // Changed from Hidden, Hidden to Feature, Hidden
        Seq(0.7f, 0.8f),
        Seq(0.9f, 1.0f)
      )),
      bias = Tensor1[Hidden](Seq(-0.1f, 0.2f))
    ),
    outputWeight = Tensor2[Hidden, Output](Seq(
      Seq(1.1f),
      Seq(1.2f)
    )),
    outputBias = Tensor1[Output](Seq(0.5f)),
    globalScale = Tensor0(2.0f)
  )
  
  println(s"Multi-layer model created with nested structure")
  println(s"  layer1 weight shape: ${multiLayerModel.layer1.weight.shape.dims}")
  println(s"  layer2 weight shape: ${multiLayerModel.layer2.weight.shape.dims}")
  println(s"  output weight shape: ${multiLayerModel.outputWeight.shape.dims}")
  println(s"  global scale: ${multiLayerModel.globalScale}")
  println()

  // 3. Using PyTree with autodiff on tuple-based structures
  println("3. PyTree with Autodiff")
  
  // Define a simple neural network forward pass using 3-tuples (which have PyTree instances)
  def neuralNetworkLoss(params: (Tensor2[Feature, Hidden], Tensor1[Hidden], Tensor0)): Tensor0 =
    val (w1, b1, scale) = params
    
    // Simplified forward pass for demonstration
    val hiddenActivation = w1.sum + b1.sum  // Simplified
    val output = hiddenActivation * scale  // Simplified
    
    // Simple loss: squared output
    output * output
  
  val networkParams = (
    Tensor2[Feature, Hidden](Seq(Seq(0.1f, 0.2f), Seq(0.3f, 0.4f))),  // w1
    Tensor1[Hidden](Seq(0.1f, 0.0f)),                                    // b1
    Tensor0(2.0f)                                                        // scale
  )
  
  val lossFunction = neuralNetworkLoss
  val gradFunction = Autodiff.grad(lossFunction)
  
  val loss = lossFunction(networkParams)
  val gradients = gradFunction(networkParams)
  
  println(s"Network loss: ${loss.toFloat}")
  println(s"Gradients computed for all parameters:")
  println(s"  ∇w1 shape: ${gradients._1.shape.dims}")
  println(s"  ∇b1 shape: ${gradients._2.shape.dims}")
  println(s"  ∇scale: ${gradients._3.toFloat}")
  println()

  // 4. Working with different parameter groupings
  println("4. Different Parameter Groupings")
  
  // Simple 2-tuple (supported by our PyTree instances)
  type SimplePair = (Tensor2[Feature, Hidden], Tensor0)
  
  def modelWithSimpleParams(params: SimplePair): Tensor0 =
    val (weight, scale) = params
    weight.sum * scale
  
  val simpleParams: SimplePair = (
    Tensor2[Feature, Hidden](Seq(Seq(1.0f, 2.0f))),
    Tensor0(2.0f)
  )
  
  val simpleLoss = modelWithSimpleParams(simpleParams)
  val simpleGrads = Autodiff.grad(modelWithSimpleParams)(simpleParams)
  
  println(s"Simple parameters loss: ${simpleLoss.toFloat}")
  println(s"Simple gradients computed successfully")
  println()

  // 5. Best practices summary
  println("5. PyTree Best Practices")
  println("✓ Use tuples for simple parameter groupings (2-3 tensors)")
  println("✓ Use nested tuples for hierarchical organization")
  println("✓ Keep PyTree structures consistent throughout training")
  println("✓ Consider parameter initialization and optimization when designing structures")
  println("✓ Use meaningful names via case classes when structure is complex")
  println("✓ Test PyTree conversion round-trip: toPyTree -> fromPyTree")
  println()
  
  // 6. Testing PyTree round-trip conversion
  println("6. Testing PyTree Round-trip")
  
  val testParams = (
    Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f)),
    Tensor2[Feature, Hidden](Seq(Seq(4.0f, 5.0f), Seq(6.0f, 7.0f))),
    Tensor0(8.0f)
  )
  
  // The conversion happens automatically in autodiff, but let's verify the types work
  println(s"Original parameters:")
  println(s"  tensor1: ${testParams._1}")
  println(s"  tensor2: ${testParams._2}")
  println(s"  scalar: ${testParams._3}")
  
  // Test that we can use these parameters with autodiff
  def testFunction(params: (Tensor1[Feature], Tensor2[Feature, Hidden], Tensor0)): Tensor0 =
    val (t1, t2, s) = params
    t1.sum + t2.sum + s
  
  val testLoss = testFunction(testParams)
  val testGrads = Autodiff.grad(testFunction)(testParams)
  
  println(s"Test function result: ${testLoss.toFloat}")
  println(s"Gradients computed successfully - PyTree conversion working!")
  println()

  println("=== Custom PyTree Complete! ===")
