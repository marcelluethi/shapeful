package examples.vae

import shapeful.*
import shapeful.autodiff.*
import shapeful.nn.*
import shapeful.jax.{Jax, Jit}
import shapeful.random.Random
import shapeful.optimization.GradientDescent
import shapeful.tensor.TensorIndexing.*
import examples.datautils.{MNISTLoader, DataLoaderOps}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File

/** Simple Variational Autoencoder (VAE) implementation using Shapeful.
  *
  * This is a simplified example demonstrating:
  *   - Basic encoder-decoder architecture
  *   - Reparameterization trick
  *   - VAE loss computation
  *   - Gradient-based training with JIT compilation
  */
object VariationalAutoEncoderMNIST:
  // Type aliases for dimensionality labels
  type Sample = "sample"
  type Feature = "feature"
  given Dim[Feature] = Dim(784)

  type Hidden1 = "hidden1"
  given Dim[Hidden1] = Dim(512)

  type Hidden2 = "hidden2"
  given Dim[Hidden2] = Dim(256)

  type Latent = "latent"
  given Dim[Latent] = Dim(20)

  type Height = "height"
  type Width = "width"

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
  ) derives TensorTree,
        ToPyTree

  // Initialize parameters with multi-layer architecture
  def initParams(key: Random.Key): VAEParams =
    val keys = key.split(7)
    VAEParams(
      // Encoder layers with He initialization for ReLU
      encoder_layer1 = Linear.he[Feature, Hidden1](keys(0)),
      encoder_layer2 = Linear.he[Hidden1, Hidden2](keys(1)),
      encoder_mean = Linear.xavier[Hidden2, Latent](keys(2)),
      encoder_logvar = Linear.xavier[Hidden2, Latent](keys(3)),

      // Decoder layers with He initialization for ReLU
      decoder_layer1 = Linear.he[Latent, Hidden2](keys(4)),
      decoder_layer2 = Linear.he[Hidden2, Hidden1](keys(5)),
      decoder_output = Linear.xavier[Hidden1, Feature](keys(6))
    )

  // Multi-layer encoder with ReLU activations
  def encode(params: VAEParams, x: Tensor1[Feature]): (Tensor1[Latent], Tensor1[Latent]) =
    val layer1 = Linear[Feature, Hidden1]()
    val layer2 = Linear[Hidden1, Hidden2]()
    val meanLayer = Linear[Hidden2, Latent]()
    val logvarLayer = Linear[Hidden2, Latent]()

    // Forward pass through encoder
    val h1 = Activation.relu(layer1(params.encoder_layer1)(x))
    val h2 = Activation.relu(layer2(params.encoder_layer2)(h1))
    val mean = meanLayer(params.encoder_mean)(h2)
    val logvar = logvarLayer(params.encoder_logvar)(h2)

    (mean, logvar)

  // Reparameterization trick - takes epsilon as input instead of generating it
  def reparameterize(mean: Tensor1[Latent], logvar: Tensor1[Latent], eps: Tensor1[Latent]): Tensor1[Latent] =
    val std = (logvar * Tensor0(0.5f)).exp
    mean + (eps * std)

  // Multi-layer decoder with ReLU activations
  def decode(params: VAEParams, z: Tensor1[Latent]): Tensor1[Feature] =
    val layer1 = Linear[Latent, Hidden2]()
    val layer2 = Linear[Hidden2, Hidden1]()
    val outputLayer = Linear[Hidden1, Feature]()

    // Forward pass through decoder
    val h1 = Activation.relu(layer1(params.decoder_layer1)(z))
    val h2 = Activation.relu(layer2(params.decoder_layer2)(h1))
    val output = outputLayer(params.decoder_output)(h2)

    Activation.sigmoid(output)

  // Forward pass - takes epsilon as input
  def forward(
      params: VAEParams,
      x: Tensor1[Feature],
      eps: Tensor1[Latent]
  ): (Tensor1[Feature], Tensor1[Latent], Tensor1[Latent]) =
    val (mean, logvar) = encode(params, x)
    val z = reparameterize(mean, logvar, eps)
    val x_recon = decode(params, z)
    (x_recon, mean, logvar)

  // VAE loss for a single sample - takes epsilon as input
  def vae_loss(params: VAEParams, x: Tensor1[Feature], epsNoise: Tensor1[Latent]): Tensor0 =
    val (x_recon, mean, logvar) = forward(params, x, epsNoise)

    // binary cross entropy
    val eps = Tensor0(1e-8f) // Small epsilon to avoid log(0)
    val x_recon_clipped = x_recon.clamp(eps, Tensor0(1.0f) - eps)
    val bce_terms = x * x_recon_clipped.log + (Tensor.ones(x.shape) - x) * (Tensor.ones(x.shape) - x_recon_clipped).log
    val recon_loss = bce_terms.sum * Tensor0(-1.0f)

    // // KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    val mean_sq = mean * mean
    val logvar_exp = logvar.exp
    val ones = Tensor.ones(logvar.shape)
    val kl_terms = ones + logvar - mean_sq - logvar_exp
    val kl_loss = kl_terms.sum * Tensor0(-0.5f)

    recon_loss + kl_loss

  // Batch loss using vmap - takes batch of epsilon values as input
  def batchLoss(batchImages: Tensor2[Sample, Feature], batchEps: Tensor2[Sample, Latent])(params: VAEParams): Tensor0 =
    // Compute loss for each sample using zipVmap
    val losses = batchImages.zipVmap(Axis[Sample])(batchEps) { (sample, epsNoise) =>
      vae_loss(params, sample, epsNoise)
    }
    shapeful.mean(losses)

  // Function to save a 784-dimensional tensor as a 28x28 PNG image
  def saveTensorAsPNG(tensor: Tensor1[Feature], filename: String): Unit =
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
    for i <- scalaArray.indices do
      val row = i / width // Row index (0-27)
      val col = i % width // Column index (0-27)

      // Clamp values to [0, 1] range (MNIST should already be normalized)
      val clampedVal = scalaArray(i).max(0.0f).min(1.0f)
      val grayValue = (clampedVal * 255).toInt

      // Set pixel (BufferedImage coordinates: x=col, y=row)
      image.setRGB(col, row, (grayValue << 16) | (grayValue << 8) | grayValue)

    // Save the image
    val file = new File(filename)
    javax.imageio.ImageIO.write(image, "PNG", file)
    println(s"Saved image to $filename (${width}x${height})")

  @main def runVAEMNIST(): Unit =
    println("=== Multi-layer Variational Autoencoder Example ===\n")

    val batchSize = 128 // Batch size
    val learningRate = 0.002f // Learning rate
    val numEpochs = 10 // Number of epochs

    println(s"VAE Configuration:")
    println(s"- Feature size: 784 (MNIST 28x28)")
    println(s"- Hidden layer 1: 512")
    println(s"- Hidden layer 2: 256")
    println(s"- Latent size: 20")
    println(s"- Batch size: $batchSize")
    println(s"- Learning rate: $learningRate")
    println(s"- Epochs: $numEpochs")

    // Initialize parameters
    val key = Random.Key(42)
    val (initKey, trainKey) = key.split2()
    val params = initParams(initKey)
    println("\nInitialized multi-layer VAE parameters")

    // Load MNIST dataset
    println("\nLoading MNIST dataset...")
    val dataset = MNISTLoader.createTrainingDataset(maxSamples = None).get
    println(s"Loaded dataset with ${dataset.size} images")

    // Split dataset for train/validation
    val (trainData, validData) = DataLoaderOps.split(dataset, 0.9)
    println(s"Training samples: ${trainData.size}, Validation samples: ${validData.size}")

    var currentParams = params
    var rngKey = trainKey

    // JIT-compile the entire gradient step (forward + backward + parameter update)
    // This is significantly faster than JITing just the loss function
    val jittedGradStep = Jit.gradientStep(
      (params: VAEParams, flattenedImages: Tensor2[Sample, Feature], batchEps: Tensor2[Sample, Latent]) =>
        val gradFn = Autodiff.grad((p: VAEParams) => batchLoss(flattenedImages, batchEps)(p))
        GradientDescent(learningRate).step(gradFn, params)
    )

    // Also JIT the loss computation for evaluation/reporting (doesn't update params)
    val jittedBatchLoss = Jit.apply[
      (VAEParams, Tensor2[Sample, Feature], Tensor2[Sample, Latent]),
      EmptyTuple
    ] { case (params, batchImages, batchEps) =>
      batchLoss(batchImages, batchEps)(params)
    }

    // Training loop with JIT-compiled gradient steps
    println("\nTraining VAE with JIT-compiled gradient steps...")
    for epoch <- 0 until numEpochs do
      println(s"\nEpoch ${epoch + 1}/$numEpochs")
      System.gc() // Suggest garbage collection to manage memory
      var batchCount = 0

      // Iterate over batches
      trainData.batches[Sample](batchSize).zipWithIndex.foreach { case ((batchImages, batchLabels), batchIdx) =>
        val actualBatchSize = batchImages.shape.dim[Sample]

        if actualBatchSize == batchSize then
          // Flatten images from (Sample, Height, Width) to (Sample, Feature)
          val flattenedImages = batchImages.vmap(Axis[Sample]) { image => image.reshape(Shape(Axis[Feature] -> 784)) }

          // Generate random epsilon samples OUTSIDE JIT
          val (batchKey, nextKey) = rngKey.split2()
          rngKey = nextKey
          val batchEps = Tensor.randn(batchKey, Shape(Axis[Sample] -> batchSize, Axis[Latent] -> 20))

          // Apply JIT-compiled gradient step (forward + backward + update in one compiled function)
          currentParams = jittedGradStep(currentParams, flattenedImages, batchEps)

          batchCount += 1

          // Compute and report loss every 50 batches
          if batchIdx % 50 == 0 then
            val loss = jittedBatchLoss((currentParams, flattenedImages, batchEps))
            val lossValue = Jax.jnp.array(loss.jaxValue).item().as[Float]
            println(f"  Batch $batchIdx - Loss: $lossValue%.4f")
      }

      println(s"  Processed $batchCount batches")

    println("\nTraining complete!")

    // Create output directory
    val outputDir = "vae_output"
    new File(outputDir).mkdirs()

    // Test reconstruction on validation samples
    println("\nTesting reconstruction...")
    val (testImages3D, testLabels) = validData.getBatch[Sample](0, 5)

    for idx <- 0 until 5 do
      // testImages3D is Tensor3[Sample, Height, Width], we need to get one image and flatten it
      val testImage2D = testImages3D.slice[Sample](idx, idx + 1).reshape(Shape(Axis[Height] -> 28, Axis[Width] -> 28))
      val flattenedImage = testImage2D.reshape(Shape(Axis[Feature] -> 784))

      val testKey = Random.Key(999 + idx)
      val eps = Tensor.randn(testKey, Shape(Axis[Latent] -> 20))
      val (reconstructed, mean, logvar) = forward(currentParams, flattenedImage, eps)

      // Save original and reconstructed images
      saveTensorAsPNG(flattenedImage, s"$outputDir/original_$idx.png")
      saveTensorAsPNG(reconstructed, s"$outputDir/reconstructed_$idx.png")

      if idx == 0 then
        println(f"\nReconstruction test for sample $idx:")
        println(f"Mean latent shape:   ${mean.shape}")
        println(f"Logvar latent shape: ${logvar.shape}")

    // Generate and save samples from prior
    println(f"\nGenerating samples from prior...")
    var genKey = trainKey.split2()._1
    for i <- 0 until 5 do
      val keys = genKey.split2()
      genKey = keys._2
      val priorSample = Tensor.randn(keys._1, Shape(Axis[Latent] -> 20))
      val generated = decode(currentParams, priorSample)
      saveTensorAsPNG(generated, s"$outputDir/generated_$i.png")

    println(f"\nAll images saved to '$outputDir/' directory")
