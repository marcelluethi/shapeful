// package examples.basic

// import scala.language.experimental.namedTypeArguments
// import shapeful.*
// import shapeful.autodiff.*
// import shapeful.nn.*
// import shapeful.jax.Jax
// import shapeful.random.Random
// import shapeful.optimization.GradientDescent
// import shapeful.nn.Layer.LayerDim
// import examples.DataUtils
// import examples.nn.losses.BinaryCrossEntropy

// object MLPClassifier {

//   type Sample = "sample"

//   type Feature = "feature"
//   given LayerDim[Feature] = LayerDim(2)
 
//   type Hidden1 = "hidden1"
//   given LayerDim[Hidden1] = LayerDim(10)

//   type Output = "output"
//   given LayerDim[Output] = LayerDim(2)
//     def dim: Int = 2
    
//   case class MLPParams(
//     layer1: Linear.Params[Feature, Hidden1],
//     output: Linear.Params[Hidden1, Output]
//   ) derives TensorTree, ToPyTree
    
//   def initParams(key: Random.Key): MLPParams = {
//       val keys = key.split(2)
//       MLPParams(
//           layer1 = Linear.he[Feature, Hidden1](keys(0)),
//           output = Linear.xavier[Hidden1, Output](keys(1))
//       )
//   }

//   def forward(params: MLPParams, x: Tensor1[Feature]): (Tensor1[Output], Tensor1[Output]) = {
//     val layer1 = Linear[Feature, Hidden1]()
//     val outputLayer = Linear[Hidden1, Output]()
    
//     val mapping = layer1(params.layer1)
//     .andThen(Activation.relu)
//     .andThen(outputLayer(params.output))

//     val logits = mapping(x)
//     val probs = Activation.softmax[SoftmaxAxis=Output](logits)
//     (logits, probs)
//   }
  
//   def main(args: Array[String]): Unit = {

//     val learningRate = 5e-1f
//     val numSamples = 1000
//     val key = Random.Key(42)

//     val (dataKey, trainKey) = Random.Key(42).split2()
//     val (trainingData, labels) = DataUtils.twoMoons[Sample, Feature, Sample](numSamples, dataKey)

//     val (initKey, restKey) = trainKey.split2()
//     val (lossKey, sampleKey) = restKey.split2()

//     val labelsOneHot = Utils.oneHot[Classes=Output](labels, 2)

//     def  loss(p: MLPParams) : Tensor0 =  {
//       val losses = trainingData.zipVmap[VmapAxis = Sample](labelsOneHot)((sample, label) => 
//         val (logits, _) = forward(p, sample)
//         BinaryCrossEntropy(logits, label)
//       )
//       losses.mean
//     }
    
//     val initialParams = initParams(initKey)
      
//     val gradFn = Autodiff.grad(loss)
//     val finalParams = GradientDescent(learningRate).optimize(gradFn, initialParams)
//     .zipWithIndex
//     .map((params, i) => 
//       if i % 100 == 0 then
//           println(loss(params))
//           val outputs = trainingData.vmap[VmapAxis = Sample](x => forward(params, x)._2)
//           val labelsOneHot = Utils.oneHot[Classes=Output](labels, 2)
//           println("acc: " + (outputs - labelsOneHot).abs.sum)
//       end if
//       params
//     )
//     .take(2500).toSeq.last

//     val predictions = trainingData.vmap[VmapAxis = Sample](x => forward(finalParams, x)._2)
//     println(predictions)
//     val predictionClasses = predictions.vmap[VmapAxis = Sample](p => p.argmax)

//     println("\nTraining complete. Optimized parameters:" + finalParams)
 

//   }
// }

