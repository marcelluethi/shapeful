package examples.vae

import shapeful.*
// import shapeful.autodiff.*
// import shapeful.nn.*
// import shapeful.jax.Jax
// import shapeful.random.Random
// import java.awt.image.BufferedImage
// import javax.imageio.ImageIO
// import java.io.File
// import shapeful.distributions.MVNormal
// import examples.autodiff.LinearRegression.Sample
// import java.nio.file.{Files, Paths}
// import shapeful.optimization.GradientDescent
// import scaltair.Chart
// import scaltair.Channel
// import scaltair.FieldType
// import scaltair.Mark
// import examples.DataUtils

// /**
//  * Simple Variational Autoencoder (VAE) implementation using Shapeful.
//  *
//  * This is a simplified example demonstrating:
//  * - Basic encoder-decoder architecture
//  * - Reparameterization trick
//  * - VAE loss computation
//  * - Gradient-based training
//  */
// object VariationalAutoEncoder {
//   // Type aliases for dimensionality labels
//   type Batch = "batch"
//   type Feature = "feature"
//   type Hidden1 = "hidden1"
//   type Hidden2 = "hidden2"
//   type Latent = "latent"
//   type VmapAxis = "sample"

//   case class VAEParams(
//     // Encoder parameters
//     encoder_layer1: Linear.Params[Feature, Hidden1],
//     //encoder_layer2: Linear.Params[Hidden1, Hidden2],
//     encoder_mean: Linear.Params[Hidden1, Latent],
//     encoder_logvar: Linear.Params[Hidden1, Latent],

//     // Decoder parameters
//     decoder_layer1: Linear.Params[Latent, Hidden1],
//     decoder_layer2: Linear.Params[Hidden1, Hidden2],
//     decoder_output: Linear.Params[Hidden2, Feature]
//   ) derives TensorTree, ToPyTree

//   // Initialize parameters with multi-layer architecture
//   def initParams(inputSize: Int, encHiddenSize: Int, decHiddenSize1: Int, decHiddenSize2: Int, latentSize: Int, key: Random.Key): VAEParams = {
//       val keys = key.split(7)
//       val p = VAEParams(
//           encoder_layer1 = Linear.he[Feature, Hidden1](inputSize, encHiddenSize, keys(0)),
//           //encoder_layer2 = Linear.he[Hidden1, Hidden2](hiddenSize1, hiddenSize2, keys(1)),
//           encoder_mean = Linear.xavier[Hidden1, Latent](encHiddenSize, latentSize, keys(2)),
//           encoder_logvar = Linear.Params[Hidden1, Latent](
//             weight = Tensor.zeros(Shape2[Hidden1, Latent](encHiddenSize, latentSize)),
//             bias = Tensor.zeros(Shape1[Latent](latentSize))
//           ),
//           decoder_layer1 = Linear.he[Latent, Hidden1](latentSize, decHiddenSize1, keys(3)),
//           decoder_layer2 = Linear.he[Hidden1, Hidden2](decHiddenSize1, decHiddenSize2, keys(5)),
//           decoder_output = Linear.xavier[Hidden2, Feature](decHiddenSize2, inputSize, keys(6))
//       )
//     p
//   }

//   // Multi-layer encoder with ReLU activations
//   def encode(params: VAEParams, x: Tensor1[Feature]): (Tensor1[Latent], Tensor1[Latent]) = {
//     val layer1 = Linear[Feature, Hidden1]()
//     //val layer2 = Linear[Hidden1, Hidden2]()
//     val meanLayer = Linear[Hidden1, Latent]()
//     val logvarLayer = Linear[Hidden1, Latent]()
//     val relu1 = Activation.ReLu[Tuple1[Hidden1]]()
//     val relu2 = Activation.ReLu[Tuple1[Hidden1]]()

//     // Forward pass through encoder
//     val h1 = relu1(layer1(params.encoder_layer1)(x))
//    // val h2 = relu2(layer2(params.encoder_layer2)(h1))
//     val mean = meanLayer(params.encoder_mean)(h1)
//     val logvar = logvarLayer(params.encoder_logvar)(h1)
//     // //val logvar = logvar_raw.clamp(-5.0f, 5.0f)  // Reasonable variance range

//     (mean, logvar)
//   }

//   // Reparameterization trick
//   def reparameterize(mean: Tensor1[Latent], logvar: Tensor1[Latent], key: Random.Key): Tensor1[Latent] = {
//     val std = logvar.exp.sqrt
//     val eps = Tensor.randn(mean.shape, key = Random.Key.random()) // TODO how to do this better?
//     mean + (eps * std)
//   }

//   // Multi-layer decoder with ReLU activations
//   def decode(params: VAEParams, z: Tensor1[Latent]): Tensor1[Feature] = {
//     val layer1 = Linear[Latent, Hidden1]()
//     val layer2 = Linear[Hidden1, Hidden2]()
//     val outputLayer = Linear[Hidden2, Feature]()
//     val relu1 = Activation.ReLu[Tuple1[Hidden1]]()
//     val relu2 = Activation.ReLu[Tuple1[Hidden2]]()

//     // Forward pass through decoder
//     val h1 = relu1(layer1(params.decoder_layer1)(z))
//     val h2 = relu2(layer2(params.decoder_layer2)(h1))
//     val output = outputLayer(params.decoder_output)(h2)
//     output
//   }

//   // Forward pass
//   def forward(params: VAEParams, x: Tensor1[Feature], key: Random.Key): Tensor1[Feature] = {
//     val (mean, logvar) = encode(params, x)

//     val z = reparameterize(mean, logvar, key)
//     //val (_, z) = encode(params, x)

//     val reconstructed = decode(params, z)
//     //(reconstructed, mean, logvar)
//     reconstructed
//   }

//   def vae_loss(params: VAEParams, x: Tensor1[Feature], beta : Tensor0, key: Random.Key, numMCSamples: Int = 5): Tensor0 = {
//     val (mean, logvar) = encode(params, x)
//     // val (_, z) = encode(params, x)
//     // val xRecon = decode(params, z)
//     // ((xRecon - x).dot(xRecon - x))

//     // Monte Carlo estimation of E_{q(z|x)}[-log p(x|z)] using vmapSample for parallel execution
//     val reconstruction_expectation = {
//       // Use vmapSample for efficient parallel Monte Carlo sampling
//       val samples = Random.vmapSample(key, numMCSamples, (k: Random.Key) => {

//         val z = reparameterize(mean, logvar, k)
//         val x_recon = decode(params, z)

//         val mse = ((x_recon - x).dot(x_recon - x))
//         mse
//       })

//       // Average over all Monte Carlo samples
//       samples.mean
//     }

//     // Analytical KL divergence: KL[q(z|x) || p(z)] where p(z) = N(0,I)
//     val kl_divergence = {

//       val traceSigma = logvar.exp.sum
//       val muDotMu = mean.dot(mean)
//       val Dz = Tensor0(mean.shape.size.toFloat)
//       val logdet = logvar.sum
//       Tensor0(0.5f) * (traceSigma + muDotMu - Dz - logdet)
//     }

//     reconstruction_expectation  + kl_divergence * beta

//   }

//   def main(args: Array[String]): Unit = {

//     def scatter(data : Map[String, Seq[Double | Int]], filename : String) : Unit = {
//       Chart(data)
//         .encode(
//           Channel.X("x", FieldType.Quantitative),
//           Channel.Y("y", FieldType.Quantitative),
//           Channel.Color("label", FieldType.Nominal)
//         )
//         .mark(Mark.Circle())
//         .saveHTML(filename)
//     }

//     val numSamples = 1000
//     val featureSize = 2
//     val enchiddenSize1 = 8
//     val dechiddenSize1 = 8
//     val dechiddenSize2 = 16
//     val latentSize = 2
//     val learningRate = 1e-2f
//     val beta = Tensor0(0.1f)
//     val key = Random.Key(42)

//     val (dataKey, trainKey) = Random.Key(42).split2()
//     val (trainingData, label) = DataUtils.twoMoons[Sample, Feature, "label"](numSamples, dataKey)

//     // Plot using scaltair (commented out due to missing Chart dependency)
//     val data = Map(
//       "x" -> trainingData.unstack[Sample].map(_.at(Tuple1(0)).get.toFloat.toDouble),
//       "y" -> trainingData.unstack[Sample].map(_.at(Tuple1(1)).get.toFloat.toDouble),
//       "label" -> label.unstack["label"].map(_.toFloat.toInt)
//     )

//     scatter(data, "vae_output/original.html")

//     // Initialize parameters first to get keys
//     val (initKey, restKey) = trainKey.split2()
//     val (lossKey, sampleKey) = restKey.split2()

//     def loss(beta: Tensor0, lossKey: Random.Key) = (p: VAEParams) => {
//       val losses = trainingData.vmap[VmapAxis = Sample](sample =>
//         vae_loss(p, sample, beta, numMCSamples = 1, key = Random.Key.random())
//       )
//       losses.mean
//     }

//     // data batched
//     val batchSize = 64

//     val initialParams = initParams(featureSize, enchiddenSize1, dechiddenSize1, dechiddenSize2,latentSize, initKey)
//     val gradFn = Autodiff.grad(loss(beta, lossKey))

//     val finalParams = GradientDescent(learningRate).optimize(gradFn, initialParams)
//     .zipWithIndex.map((params, i) =>
//       if i % 100 == 0 then
//         //println(params)
//           println(loss(beta, lossKey)(params))
//       end if
//       params
//     )
//     .take(2000).toSeq.last

//     // project samples into latent space

//     // check reconstruction of the original samples

//     val latentSamples = trainingData.vmap[VmapAxis = Sample](x => {
//       val (mean, logvar) = encode(finalParams, x)
//       val variance = logvar.exp
//       val latentdim = mean.shape.dims.head
//       val cov  = Tensor2.fromDiag(variance)
//       val mvn = MVNormal[Latent](mean, cov)
//       mvn.sample[Sample](1, Random.Key.random()).unstack[Sample].head
//     })

//     val latentData = Map(
//       "x" -> latentSamples.unstack[Sample].map(_.at(Tuple1(0)).get.toFloat.toDouble),
//       "y" -> latentSamples.unstack[Sample].map(_.at(Tuple1(1)).get.toFloat.toDouble)
//     )
//     scatter(latentData, "vae_output/latent.html")

//     // check reconstruction of the original samples
//     val xReconSamples = trainingData.vmap[VmapAxis = Sample](x => {
//       val recon = forward(finalParams, x, sampleKey)
//       recon
//     })

//     val reconData = Map(
//       "x" -> xReconSamples.unstack[Sample].map(_.at(Tuple1(0)).get.toFloat.toDouble),
//       "y" -> xReconSamples.unstack[Sample].map(_.at(Tuple1(1)).get.toFloat.toDouble)
//     )
//     scatter(reconData, "vae_output/reconstructed.html")

//     val zSamples = MVNormal.standardNormal(Shape1[Latent](latentSize)).sample[Sample](1000, sampleKey)
//     val xSamples = zSamples.vmap[VmapAxis = Sample](z => decode(finalParams, z))
//     val sampleData = Map(
//       "x" -> xSamples.unstack[Sample].map(_.at(Tuple1(0)).get.toFloat.toDouble),
//       "y" -> xSamples.unstack[Sample].map(_.at(Tuple1(1)).get.toFloat.toDouble)
//       )
//     scatter(sampleData, "vae_output/sampled.html")

//     println("VAE training completed successfully!")
//     println("Generated samples using vmapSample for efficient Monte Carlo estimation")
//   }
// }
