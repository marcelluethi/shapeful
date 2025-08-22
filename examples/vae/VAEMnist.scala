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

/**
 * Simple Variational Autoencoder (VAE) implementation using Shapeful.
 * 
 * This is a simplified example demonstrating:
 * - Basic encoder-decoder architecture
 * - Reparameterization trick
 * - VAE loss computation
 * - Gradient-based training
 */
object VariationalAutoEncoderMNIST {
  // Type aliases for dimensionality labels
  type Batch = "batch"
  type Feature = "feature"
  type Hidden1 = "hidden1"
  type Hidden2 = "hidden2"
  type Latent = "latent"
  
  // Multi-layer VAE parameters
  case class VAEParams(
    // Encoder layers
    encoder_layer1: Linear.Params[Feature, Hidden1],
    encoder_layer2: Linear.Params[Hidden1, Hidden2],
    encoder_mean: Linear.Params[Hidden2, Latent],
    encoder_logvar: Linear.Params[Hidden2, Latent],
    
    // Decoder layers
    decoder_layer1: Linear.Params[Latent, Hidden2],
    decoder_layer2: Linear.Params[Hidden2, Hidden1],
    decoder_output: Linear.Params[Hidden1, Feature]
  ) derives TensorTree, ToPyTree
  
  // Initialize parameters with multi-layer architecture
  def initParams(inputSize: Int, hiddenSize1: Int, hiddenSize2: Int, latentSize: Int): VAEParams = {
    VAEParams(
      // Encoder layers with He initialization for ReLU
      encoder_layer1 = Linear.he[Feature, Hidden1](inputSize, hiddenSize1),
      encoder_layer2 = Linear.he[Hidden1, Hidden2](hiddenSize1, hiddenSize2),
      encoder_mean = Linear.xavier[Hidden2, Latent](hiddenSize2, latentSize),
      encoder_logvar = Linear.xavier[Hidden2, Latent](hiddenSize2, latentSize),
      
      // Decoder layers with He initialization for ReLU
      decoder_layer1 = Linear.he[Latent, Hidden2](latentSize, hiddenSize2),
      decoder_layer2 = Linear.he[Hidden2, Hidden1](hiddenSize2, hiddenSize1),
      decoder_output = Linear.xavier[Hidden1, Feature](hiddenSize1, inputSize)
    )
  }
  
  // Multi-layer encoder with ReLU activations
  def encode(params: VAEParams, x: Tensor1[Feature]): (Tensor1[Latent], Tensor1[Latent]) = {
    val layer1 = Linear[Feature, Hidden1]()
    val layer2 = Linear[Hidden1, Hidden2]()
    val meanLayer = Linear[Hidden2, Latent]()
    val logvarLayer = Linear[Hidden2, Latent]()
    val relu = Activation.ReLu[Tuple1[Hidden1]]()
    val relu2 = Activation.ReLu[Tuple1[Hidden2]]()
    
    // Forward pass through encoder
    val h1 = relu(layer1(params.encoder_layer1)(x))
    val h2 = relu2(layer2(params.encoder_layer2)(h1))
    val mean = meanLayer(params.encoder_mean)(h2)
    val logvar = logvarLayer(params.encoder_logvar)(h2)
    
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
    val layer1 = Linear[Latent, Hidden2]()
    val layer2 = Linear[Hidden2, Hidden1]()
    val outputLayer = Linear[Hidden1, Feature]()
    val relu = Activation.ReLu[Tuple1[Hidden2]]()
    val relu2 = Activation.ReLu[Tuple1[Hidden1]]()
    
    // Forward pass through decoder
    val h1 = relu(layer1(params.decoder_layer1)(z))
    val h2 = relu2(layer2(params.decoder_layer2)(h1))
    val output = outputLayer(params.decoder_output)(h2)
    
    output.sigmoid
  }
  
  // Forward pass
  def forward(params: VAEParams, x: Tensor1[Feature], key: Int): (Tensor1[Feature], Tensor1[Latent], Tensor1[Latent]) = {
    val (mean, logvar) = encode(params, x)
    val z = reparameterize(mean, logvar, key)
    val x_recon = decode(params, z)
    (x_recon, mean, logvar)
  }
  

  def vae_loss(params: VAEParams, x: Tensor1[Feature], key: Int): Tensor0 = {
    val (x_recon, mean, logvar) = forward(params, x, key)

    // binary cross entropy  
    val eps = Tensor0(1e-8f) // Small epsilon to avoid log(0)
    val x_recon_clipped = x_recon.clamp(eps.toFloat, 1.0f - eps.toFloat)
    val bce_terms = x * x_recon_clipped.log + (Tensor.ones(x.shape) - x) * (Tensor.ones(x.shape) - x_recon_clipped).log
    val recon_loss = bce_terms.sum * Tensor0(-1.0f)
    
    // KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    val mean_sq = mean * mean
    val logvar_exp = logvar.exp
    val ones = Tensor.ones(logvar.shape)
    val kl_terms = ones + logvar - mean_sq - logvar_exp
    val kl_loss = kl_terms.sum * Tensor0(-0.5f)
    
    recon_loss + kl_loss
  }
  
  
  // Function to save a 784-dimensional tensor as a 28x28 PNG image
  def saveTensorAsPNG(tensor: Tensor1[Feature], filename: String): Unit = {
    // Convert JAX tensor to Python array for easier access
    val pythonArray = Jax.jnp.array(tensor.jaxValue).tolist()
    val scalaArray = pythonArray.as[Seq[Float]].toArray
    val width = 28
    val height = 28
    
    // Create BufferedImage
    val image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    
    // Debug: print some values to understand the data
    println(s"Tensor data length: ${scalaArray.length}")
    println(s"First 10 values: ${scalaArray.take(10).mkString(", ")}")
    println(s"Min/max values: ${scalaArray.min} / ${scalaArray.max}")
    
    // Convert tensor data to image pixels
    // The tensor is flattened in row-major order: [row0_col0, row0_col1, ..., row0_col27, row1_col0, ...]
    for (i <- scalaArray.indices) {
      val row = i / width  // Row index (0-27)
      val col = i % width  // Column index (0-27)
      
      // Clamp values to [0, 1] range (MNIST should already be normalized)
      val clampedVal = scalaArray(i).max(0.0f).min(1.0f)
      val grayValue = (clampedVal * 255).toInt
      
      // Set pixel (BufferedImage coordinates: x=col, y=row)
      image.setRGB(col, row, (grayValue << 16) | (grayValue << 8) | grayValue)
    }
    
    // Save the image
    val file = new File(filename)
    javax.imageio.ImageIO.write(image, "PNG", file)
    println(s"Saved image to $filename (${width}x${height})")
  }




  @main def runVAEMNIST(): Unit = {
    println("=== Multi-layer Variational Autoencoder Example ===\n")
    
    val featureSize = 784   // MNIST image size (28x28 flattened)
    val hiddenSize1 = 512   // First hidden layer
    val hiddenSize2 = 256   // Second hidden layer  
    val latentSize = 20     // Latent dimension
    val batchSize = 100      // Batch size
    val learningRate = 0.001f  // Learning rate
    val numEpochs = 1    // Reduced for efficiency
    val maxImages = 1000     // Limit to 5000 images for faster training
    
    println(s"VAE Configuration:")
    println(s"- Feature size: $featureSize (MNIST 28x28)")
    println(s"- Hidden layer 1: $hiddenSize1")
    println(s"- Hidden layer 2: $hiddenSize2")
    println(s"- Latent size: $latentSize")
    println(s"- Batch size: $batchSize")
    println(s"- Learning rate: $learningRate")
    println(s"- Max images: $maxImages")
    
    // Initialize parameters
    val params = initParams(featureSize, hiddenSize1, hiddenSize2, latentSize)
    println("\nInitialized multi-layer VAE parameters")
    
    // Create MNIST dataset (lazy loading with limited images)
    println("\nCreating MNIST dataset...")
    val mnistDataset = MNISTLoader.createMNISTDatasetSafe(maxImages = maxImages)
    println(s"Created MNIST dataset with ${mnistDataset.effectiveNumImages} images")
    
    // Calculate number of batches per epoch
    val numBatches = (mnistDataset.effectiveNumImages + batchSize - 1) / batchSize
    println(s"Training with $numBatches batches per epoch")
    
    // Training loop with proper batching
    println("\nTraining VAE...")
    var currentParams = params
    
    for (epoch <- 1 to numEpochs) {
      var epochLoss = 0.0f
      var processedBatches = 0
      
      // Process all batches in the dataset
      for (batchIdx <- 0 until numBatches) {
        val startIdx = batchIdx * batchSize
        val actualBatchSize = math.min(batchSize, mnistDataset.effectiveNumImages - startIdx)
        
        if (actualBatchSize > 0) {
          // Load current batch
          val batchData = MNISTLoader.datasetToTensors[Feature](mnistDataset, startIdx, actualBatchSize)
          
          // Create gradient function for this batch
          val gradFn = Autodiff.grad((p: VAEParams) => {
            val losses = batchData.zipWithIndex.map { case (sample, idx) =>
              vae_loss(p, sample, epoch * 1000 + batchIdx * 100 + idx)
            }
            losses.reduce(_ + _) * Tensor0(1.0f / actualBatchSize.toFloat) // Average loss
          })
          
          // Compute gradients and loss
          val grads = gradFn(currentParams)
          val batchLoss = batchData.zipWithIndex.map { case (sample, idx) =>
            vae_loss(currentParams, sample, epoch * 1000 + batchIdx * 100 + idx)
          }.reduce(_ + _)
          
          // SGD update
          currentParams = currentParams.zipMap(grads, 
            [T <: Tuple] => (param: Tensor[T], grad: Tensor[T]) => param - grad * Tensor0(learningRate)
          )
          
          epochLoss += batchLoss.toFloat / actualBatchSize.toFloat
          processedBatches += 1
          
          // Print progress every 100 batches
          if (batchIdx % 10 == 0) {
            println(f"  Epoch $epoch - Batch $batchIdx/$numBatches - Batch Loss: ${batchLoss.toFloat / actualBatchSize.toFloat}%.4f")
            System.gc()
          }
        }
      }
      
      val avgLoss = epochLoss / processedBatches
      println(f"Epoch $epoch%2d/$numEpochs - Average Loss: $avgLoss%.4f")
    }
    
    println("\nTraining complete!")
    
    // Create output directory
    val outputDir = "vae_output"
    new File(outputDir).mkdirs()
    
    // Test reconstruction on a few samples
    val testBatch = MNISTLoader.datasetToTensors[Feature](mnistDataset, 0, 5)
    
    for ((testSample, idx) <- testBatch.zipWithIndex) {
      val (reconstructed, mean, logvar) = forward(currentParams, testSample, 999 + idx)
      
      // Save original and reconstructed images
      saveTensorAsPNG(testSample, s"$outputDir/original_$idx.png")
      saveTensorAsPNG(reconstructed, s"$outputDir/reconstructed_$idx.png")
      
      if (idx == 0) {
        println(f"\nReconstruction test for sample $idx:")
        println(f"Mean latent:   ${mean.jaxValue}")
        println(f"Logvar latent: ${logvar.jaxValue}")
      }
    }

    // Generate and save multiple samples from prior
    println(f"\nGenerating samples from prior...")
    for (i <- 1 to 5) {
      val priorSample = Tensor.randn(Shape1[Latent](latentSize), key = 777 + i)
      val generated = decode(currentParams, priorSample)
      saveTensorAsPNG(generated, s"$outputDir/generated_$i.png")
    }
    
    println(f"\nAll images saved to '$outputDir/' directory")
  }

}