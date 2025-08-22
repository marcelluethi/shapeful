package examples.vae

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.autodiff.*
import shapeful.nn.*
import shapeful.jax.Jax
import examples.mnist.MNISTLoader
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import shapeful.distributions.MVNormal
import examples.autodiff.LinearRegression.Sample
import scaltair.Chart
import scaltair.Channel
import scaltair.FieldType
import scaltair.Mark

import scaltair.PlotTarget
import java.nio.file.{Files, Paths}
import shapeful.optimization.GradientDescent

/**
 * Simple Variational Autoencoder (VAE) implementation using Shapeful.
 * 
 * This is a simplified example demonstrating:
 * - Basic encoder-decoder architecture
 * - Reparameterization trick
 * - VAE loss computation
 * - Gradient-based training
 */
object VariationalAutoEncoder {
  // Type aliases for dimensionality labels
  type Batch = "batch"
  type Feature = "feature"
  type Hidden1 = "hidden1"
  type Latent = "latent"
  
  // Multi-layer VAE parameters
  case class VAEParams(
    // Encoder layers
    encoder_layer1: Linear.Params[Feature, Hidden1],
    encoder_mean: Linear.Params[Hidden1, Latent],
    encoder_logvar: Linear.Params[Hidden1, Latent],
    
    // Decoder layers
    decoder_layer1: Linear.Params[Latent, Hidden1],
    decoder_output: Linear.Params[Hidden1, Feature]
  ) derives TensorTree, ToPyTree
  
  // Initialize parameters with multi-layer architecture
  def initParams(inputSize: Int, hiddenSize1: Int, latentSize: Int): VAEParams = {
    VAEParams(
      // Encoder layers with He initialization for ReLU
      encoder_layer1 = Linear.he[Feature, Hidden1](inputSize, hiddenSize1),
      encoder_mean = Linear.xavier[Hidden1, Latent](hiddenSize1, latentSize),
      encoder_logvar = Linear.xavier[Hidden1, Latent](hiddenSize1, latentSize),
      
      // Decoder layers with He initialization for ReLU
      decoder_layer1 = Linear.he[Latent, Hidden1](latentSize, hiddenSize1),
      decoder_output = Linear.xavier[Hidden1, Feature](hiddenSize1, inputSize)
    )
  }
  
  // Multi-layer encoder with ReLU activations
  def encode(params: VAEParams, x: Tensor1[Feature]): (Tensor1[Latent], Tensor1[Latent]) = {
    val layer1 = Linear[Feature, Hidden1]()
    val meanLayer = Linear[Hidden1, Latent]()
    val logvarLayer = Linear[Hidden1, Latent]()
    val relu = Activation.ReLu[Tuple1[Hidden1]]()
    
    // Forward pass through encoder
    val h1 = (layer1(params.encoder_layer1)(x))
    val mean = meanLayer(params.encoder_mean)(h1)
    val logvar = logvarLayer(params.encoder_logvar)(h1)
    
    (mean, logvar)
  }
  
  // Reparameterization trick
  def reparameterize(mean: Tensor1[Latent], logvar: Tensor1[Latent], key: Int): Tensor1[Latent] = {
    val std = (logvar * Tensor0(0.5f)).exp
    val eps = Tensor.randn(mean.shape, key = key)
    mean + eps * std
  }
  
  // Multi-layer decoder with ReLU activations
  def decode(params: VAEParams, z: Tensor1[Latent]): Tensor1[Feature] = {
    val layer1 = Linear[Latent, Hidden1]()
    val outputLayer = Linear[Hidden1, Feature]()
    val relu = Activation.ReLu[Tuple1[Hidden1]]()
    
    // Forward pass through decoder
    val h1 = (layer1(params.decoder_layer1)(z))
    val output = outputLayer(params.decoder_output)(h1)

    //output.sigmoid
    output
  }
  
  // Forward pass
  def forward(params: VAEParams, x: Tensor1[Feature], key: Int): (Tensor1[Feature], Tensor1[Latent], Tensor1[Latent]) = {
    val (mean, logvar) = encode(params, x)
    val z = reparameterize(mean, logvar, key)
    val x_recon = decode(params, z)
    (x_recon, mean, logvar)
  }
  
  def vae_loss(params: VAEParams, x: Tensor1[Feature], key: Int, numMCSamples: Int = 5): Tensor0 = {
    val (mean, logvar) = encode(params, x)
    
    // Monte Carlo estimation of E_{q(z|x)}[-log p(x|z)]
    val reconstruction_expectation = {
      val samples = (0 until numMCSamples).map { i =>
        val z = reparameterize(mean, logvar, key + i)
        val x_recon = decode(params, z)
        
        // Negative log-likelihood: -log p(x|z) assuming p(x|z) = N(decoder(z), σ²I)
        val squared_error = (x - x_recon) * (x - x_recon)
        squared_error.sum * Tensor0(0.5f) // Assuming unit variance σ² = 1
      }
      samples.reduce(_ + _) * Tensor0(1.0f / numMCSamples.toFloat)
    }
    
    // Analytical KL divergence: KL[q(z|x) || p(z)] where p(z) = N(0,I)
    val kl_divergence = {
      // val mean_sq = mean * mean
      // val logvar_exp = logvar.exp
      // val ones = Tensor.ones(logvar.shape)
      // val kl_terms = ones + mean_sq - logvar + logvar_exp
      // kl_terms.sum * Tensor0(0.5f)
      val traceSigma = Tensor.ones(logvar.shape).sum
      val muDotMu = mean.dot(mean)
      val Dz = Tensor0(mean.shape.size.toFloat)
      val logdet = logvar.sum
      Tensor0(0.5f) * (traceSigma + muDotMu - Dz - logdet)
    }
    
    reconstruction_expectation - kl_divergence
  }

  @main def runVAE(): Unit = {
    println("=== Multi-layer Variational Autoencoder Example ===\n")
    
    val featureSize = 2
    val hiddenSize1 = 2   // First hidden layer
    val latentSize = 2     // Latent dimension
    val batchSize = 100      // Batch size
    val learningRate = 0.0001f  // Learning rate
    val numEpochs = 100    // Reduced for efficiency

    
    println(s"VAE Configuration:")
    println(s"- Feature size: $featureSize ")
    println(s"- Hidden layer 1: $hiddenSize1")
    println(s"- Latent size: $latentSize")
    println(s"- Batch size: $batchSize")
    println(s"- Learning rate: $learningRate")
    

    
    // create sample data from a 2d gaussian with strongly correlated
    // features
    val numSamples = 1000
    val mean = Tensor.zeros(Shape1[Feature](2))
    val cov = Tensor2[Feature, Feature](Seq(Seq(5f, 0.9f), Seq(0.9f, 1f)))
    val trainingData = MVNormal(mean, cov).sample[Sample](100)

    
    // plot using scaltair
    val data = Map(
      "x" -> trainingData.unstack[Sample].map(_.at(Tuple1(0)).get.toFloat.toDouble),
      "y" -> trainingData.unstack[Sample].map(_.at(Tuple1(1)).get.toFloat.toDouble)
    )

   
    
    Chart(data)
      .encode(
        Channel.X("x", FieldType.Quantitative),
        Channel.Y("y", FieldType.Quantitative)
      )
      .mark(Mark.Circle())
      .saveHTML("vae_plot.html")


    // Calculate number of batches per epoch
    val numBatches = 1
    println(s"Training with $numBatches batches per epoch")
    
    // Training loop with proper batching
    println("\nTraining VAE...")

    val loss = (p: VAEParams) => {
      val losses = trainingData.vmap[VmapAxis = Sample](sample => vae_loss(p, sample, numMCSamples = 5, key = scala.util.Random.nextInt()))
      losses.mean
    }
    // Create gradient function for this batch
    val gradFn = Autodiff.grad(loss)

    // Initialize parameters
    val initialParams = initParams(featureSize, hiddenSize1, latentSize)
    println("\nInitialized multi-layer VAE parameters")
    val history = GradientDescent(learningRate).optimize(gradFn, initialParams)
      .map(p => {println("loss : " +loss(p)); p})
      .take(100).toSeq
    println("\nTraining complete. Optimized parameters:")
    
    // generate samples from the trained vae

    val zSamples = MVNormal.standardNormal(Shape1[Latent](latentSize)).sample[Sample](100)
    val xSamples = zSamples.vmap[VmapAxis=Sample](z => decode(history.last, z))

    // Plot the generated samples
    val sampleData = Map(
      "x" -> xSamples.unstack[Sample].map(_.at(Tuple1(0)).get.toFloat.toDouble),
      "y" -> xSamples.unstack[Sample].map(_.at(Tuple1(1)).get.toFloat.toDouble)
    )

    Chart(sampleData)
      .encode(
        Channel.X("x", FieldType.Quantitative),
        Channel.Y("y", FieldType.Quantitative)
      )
      .mark(Mark.Circle())
      .saveHTML("vae_samples-generated.html")
  }
}