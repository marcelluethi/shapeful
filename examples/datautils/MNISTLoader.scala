package examples.datautils

import scala.language.experimental.namedTypeArguments
import scala.annotation.experimental
import shapeful.*
import shapeful.jax.Jax
import examples.datautils.DataLoader
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}
import java.nio.file.{Files, Paths}
import scala.util.{Try, Success, Failure}

/**
 * Pure Scala MNIST data loader that reads IDX format files directly.
 * 
 * The MNIST dataset files use a specific binary format:
 * - Images: magic number (2051), number of images, rows, cols, then pixel data
 * - Labels: magic number (2049), number of labels, then label data
 */
object MNISTLoader {
  
  // Type labels for MNIST image dimensions
  type Height = "height"
  type Width = "width"
  
  /**
   * Lazy MNIST dataset that loads images on demand
   */
  @experimental
  case class MNISTDataset(
    imageFile: String,
    labelFile: String,
    numImages: Int,
    numLabels: Int,
    rows: Int = 28,
    cols: Int = 28,
    maxImages: Int = Int.MaxValue  // Maximum number of images to use
  ) extends DataLoader[Tensor2[Height, Width], Int] {
    // Effective number of images (limited by maxImages)
    val effectiveNumImages: Int = math.min(numImages, maxImages)
    val effectiveNumLabels: Int = math.min(numLabels, maxImages)
    
    // DataLoader trait implementation
    def size: Int = effectiveNumImages
    def apply(index: Int): (Tensor2[Height, Width], Int) = (getImage(index), getLabel(index))
    
    // Cache for loaded images/labels to avoid re-reading
    private val imageCache = scala.collection.mutable.Map[Int, Tensor2[Height, Width]]()
    private val labelCache = scala.collection.mutable.Map[Int, Int]()
    
    /**
     * Load a specific image by index (0-based)
     */
    def getImage(index: Int): Tensor2[Height, Width] = {
      if (index < 0 || index >= effectiveNumImages) {
        throw new IndexOutOfBoundsException(s"Image index $index out of bounds [0, $effectiveNumImages)")
      }
      
      imageCache.getOrElseUpdate(index, loadSingleImage(index))
    }
    
    /**
     * Load a specific label by index (0-based)
     */
    def getLabel(index: Int): Int = {
      if (index < 0 || index >= effectiveNumLabels) {
        throw new IndexOutOfBoundsException(s"Label index $index out of bounds [0, $effectiveNumLabels)")
      }
      
      labelCache.getOrElseUpdate(index, loadSingleLabel(index))
    }
    
    /**
     * Load a batch of images efficiently
     */
    def getBatchImages(startIdx: Int, batchSize: Int): Seq[Tensor2[Height, Width]] = {
      val endIdx = math.min(startIdx + batchSize, effectiveNumImages)
      (startIdx until endIdx).map(getImage)
    }
    
    /**
     * Load a batch of labels efficiently
     */
    def getBatchLabels(startIdx: Int, batchSize: Int): Seq[Int] = {
      val endIdx = math.min(startIdx + batchSize, effectiveNumLabels)
      (startIdx until endIdx).map(getLabel)
    }
    
    /**
     * Load a single image from file by index
     */
    private def loadSingleImage(index: Int): Tensor2[Height, Width] = {
      val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFile)))
      try {
        // Skip header (16 bytes)
        dis.skipBytes(16)
        // Skip to the specific image (each image is rows * cols bytes)
        val imageSize = rows * cols
        dis.skipBytes(index * imageSize)
        
        // Read the image data efficiently
        val imageBytes = new Array[Byte](imageSize)
        dis.readFully(imageBytes)
        
        // Convert to 2D float array efficiently
        val pixelRows = Array.ofDim[Seq[Float]](rows)
        for (row <- 0 until rows) {
          val rowPixels = Array.ofDim[Float](cols)
          for (col <- 0 until cols) {
            // Convert unsigned byte to normalized float
            val byteVal = imageBytes(row * cols + col) & 0xFF
            rowPixels(col) = byteVal / 255.0f
          }
          pixelRows(row) = rowPixels.toSeq
        }
        
        // Create Tensor2 using the standard constructor
        Tensor2[Height, Width](pixelRows.toSeq, DType.Float32)
        
      } finally {
        dis.close()
      }
    }
    
    /**
     * Load a single label from file by index
     */
    private def loadSingleLabel(index: Int): Int = {
      val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFile)))
      try {
        // Skip header (8 bytes)
        dis.skipBytes(8)
        // Skip to the specific label
        dis.skipBytes(index)
        // Read the label
        dis.readUnsignedByte()
      } finally {
        dis.close()
      }
    }
  }
  
  /**
   * Legacy data structure for compatibility
   */
  case class MNISTData(
    images: Array[Tensor2[Height, Width]], 
    labels: Array[Int]           
  )
  
  /**
   * Read 32-bit big-endian integer from DataInputStream
   */
  private def readInt(dis: DataInputStream): Int = {
    dis.readInt()
  }

  /**
   * Read dataset metadata without loading all images
   */
  def readDatasetInfo(imagesFile: String, labelsFile: String): Try[(Int, Int, Int, Int)] = Try {
    // Read image file header
    val imageDis = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesFile)))
    val (numImages, rows, cols) = try {
      val magic = imageDis.readInt()
      if (magic != 2051) {
        throw new IllegalArgumentException(s"Invalid magic number for images: $magic (expected 2051)")
      }
      val numImages = imageDis.readInt()
      val rows = imageDis.readInt()
      val cols = imageDis.readInt()
      (numImages, rows, cols)
    } finally {
      imageDis.close()
    }

    // Read label file header
    val labelDis = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsFile)))
    val numLabels = try {
      val magic = labelDis.readInt()
      if (magic != 2049) {
        throw new IllegalArgumentException(s"Invalid magic number for labels: $magic (expected 2049)")
      }
      labelDis.readInt()
    } finally {
      labelDis.close()
    }

    (numImages, numLabels, rows, cols)
  }

  /**
   * Create a lazy MNIST dataset
   */
  def createDataset(imagesFile: String, labelsFile: String, maxImages: Int = Int.MaxValue): Try[MNISTDataset] = {
    for {
      (numImages, numLabels, rows, cols) <- readDatasetInfo(imagesFile, labelsFile)
    } yield {
      if (numImages != numLabels) {
        throw new IllegalArgumentException(s"Mismatch: $numImages images vs $numLabels labels")
      }
      val effectiveImages = math.min(numImages, maxImages)
      println(s"Created lazy MNIST dataset: $effectiveImages images of size ${rows}x${cols} (limited from $numImages)")
      MNISTDataset(imagesFile, labelsFile, numImages, numLabels, rows, cols, maxImages)
    }
  }

  /**
   * Create training dataset (lazy loading)
   */
  def createTrainingDataset(dataDir: String = "data", maxImages: Int = Int.MaxValue): Try[MNISTDataset] = {
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset(imagesFile, labelsFile, maxImages)
  }

  /**
   * Create test dataset (lazy loading)
   */
  def createTestDataset(dataDir: String = "data", maxImages: Int = Int.MaxValue): Try[MNISTDataset] = {
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    createDataset(imagesFile, labelsFile, maxImages)
  }

  /**
   * Create a safe MNIST dataset with fallback to dummy data
   */
  def createMNISTDatasetSafe(dataDir: String = "data", maxImages: Int = Int.MaxValue, numDummySamples: Int = 1000): MNISTDataset = {
    createTrainingDataset(dataDir, maxImages) match {
      case Success(dataset) => 
        println(s"Successfully created lazy MNIST dataset with ${dataset.effectiveNumImages} images")
        dataset
      case Failure(e) => 
        println(s"Could not load MNIST data: ${e.getMessage}")
        println(s"Creating dummy dataset with $numDummySamples samples")
        createDummyDataset(numDummySamples)
    }
  }

  /**
   * Create a dummy dataset for testing
   */
  def createDummyDataset(numSamples: Int): MNISTDataset = {
    // Create a dummy dataset that generates random images on demand
    new MNISTDataset("", "", numSamples, numSamples, 28, 28, numSamples) {
      override def getImage(index: Int): Tensor2[Height, Width] = {
        if (index < 0 || index >= numImages) {
          throw new IndexOutOfBoundsException(s"Image index $index out of bounds [0, $numImages)")
        }
        
        // Generate deterministic random image based on index
        val random = new scala.util.Random(index + 42)
        val pixelRows = Array.fill(28) {
          Array.fill(28) {
            // Add some spatial correlation to make it look more like digits
            val base = if (random.nextFloat() < 0.3f) 0.7f else 0.1f
            math.max(0.0f, math.min(1.0f, base + random.nextGaussian().toFloat * 0.2f))
          }.toSeq
        }
        
        Tensor2[Height, Width](pixelRows.toSeq, DType.Float32)
      }
      
      override def getLabel(index: Int): Int = {
        if (index < 0 || index >= numLabels) {
          throw new IndexOutOfBoundsException(s"Label index $index out of bounds [0, $numLabels)")
        }
        
        // Generate deterministic random label based on index
        new scala.util.Random(index + 123).nextInt(10)
      }
    }
  }

  /**
   * Convert images to flattened tensors for a specific batch from dataset
   */
  def datasetToTensors[Feature <: Singleton](
    dataset: MNISTDataset, 
    startIdx: Int, 
    batchSize: Int
  ): Seq[Tensor1[Feature]] = {
    val endIdx = math.min(startIdx + batchSize, dataset.effectiveNumImages)
    (startIdx until endIdx).map { i =>
      val image2D = dataset.getImage(i)
      image2D.reshape(Shape1[Feature](image2D.shape.dims.product))
    }.toSeq
  }

  /**
   * Get a batch from dataset as flattened tensors (for models expecting 1D input)
   */
  def getBatchFromDataset[Feature <: Singleton](
    dataset: MNISTDataset, 
    batchIndex: Int, 
    batchSize: Int
  ): Seq[Tensor1[Feature]] = {
    val startIdx = batchIndex * batchSize
    datasetToTensors[Feature](dataset, startIdx, batchSize)
  }
  
  /**
   * Load MNIST images from IDX3 format file
   * Format: magic(4), numImages(4), rows(4), cols(4), pixels...
   */
  def loadImages(filename: String): Try[Array[Tensor2[Height, Width]]] = Try {
    val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    try {
      val magic = readInt(dis)
      if (magic != 2051) {
        throw new IllegalArgumentException(s"Invalid magic number for images: $magic (expected 2051)")
      }
      
      val numImages = readInt(dis)
      val rows = readInt(dis)
      val cols = readInt(dis)
      
      println(s"Loading $numImages images of size ${rows}x${cols} from $filename")
      
      val images = Array.ofDim[Tensor2[Height, Width]](numImages)
      
      for (i <- 0 until numImages) {
        // Read pixel data as 2D array
        println("here " + i)
        val pixelRows = Array.ofDim[Seq[Float]](rows)
        for (row <- 0 until rows) {
          val rowPixels = Array.ofDim[Float](cols)
          for (col <- 0 until cols) {
            // Read unsigned byte and normalize to [0, 1]
            val pixel = dis.readUnsignedByte()
            rowPixels(col) = pixel / 255.0f
          }
          pixelRows(row) = rowPixels.toSeq
        }
        
        // Create Tensor2 from 2D pixel data
        images(i) = Tensor2[Height, Width](pixelRows.toSeq, DType.Float32)
      }

      images
    } finally {
      dis.close()
    }
  }
  
  /**
   * Load MNIST labels from IDX1 format file
   * Format: magic(4), numLabels(4), labels...
   */
  def loadLabels(filename: String): Try[Array[Int]] = Try {
    val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    try {
      val magic = readInt(dis)
      if (magic != 2049) {
        throw new IllegalArgumentException(s"Invalid magic number for labels: $magic (expected 2049)")
      }
      
      val numLabels = readInt(dis)
      println(s"Loading $numLabels labels from $filename")
      
      val labels = Array.ofDim[Int](numLabels)
      for (i <- 0 until numLabels) {
        labels(i) = dis.readUnsignedByte()
      }
      
      labels
    } finally {
      dis.close()
    }
  }
  
  /**
   * Load complete MNIST dataset (images + labels)
   */
  def loadDataset(imagesFile: String, labelsFile: String): Try[MNISTData] = {
    for {
      images <- loadImages(imagesFile)
      labels <- loadLabels(labelsFile)
    } yield {
      if (images.length != labels.length) {
        throw new IllegalArgumentException(s"Mismatch: ${images.length} images vs ${labels.length} labels")
      }
      MNISTData(images, labels)
    }
  }
  
  /**
   * Load MNIST training dataset from standard file locations
   */
  def loadTrainingData(dataDir: String = "data"): Try[MNISTData] = {
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    loadDataset(imagesFile, labelsFile)
  }
  
  /**
   * Load MNIST test dataset from standard file locations
   */
  def loadTestData(dataDir: String = "data"): Try[MNISTData] = {
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    loadDataset(imagesFile, labelsFile)
  }
  
  /**
   * Convert MNIST images to flattened Shapeful tensors for a specific batch
   * Converts from Tensor2[Height, Width] to Tensor1[Feature] for compatibility with models expecting flattened input
   */
  def imagesToTensors[Feature <: Singleton](
    images: Array[Tensor2[Height, Width]], 
    startIdx: Int, 
    batchSize: Int
  ): Seq[Tensor1[Feature]] = {
    val endIdx = math.min(startIdx + batchSize, images.length)
    (startIdx until endIdx).map { i =>
      val image2D = images(i)
      // Flatten the 2D tensor to 1D for compatibility with existing models
      val flattenedJax = Jax.jnp.reshape(image2D.jaxValue, 784)
      new Tensor(Shape1[Feature](784), flattenedJax, DType.Float32)
    }.toSeq
  }
  
  /**
   * Get a batch of images as 2D Shapeful tensors (preserving image structure)
   */
  def getBatch2D(
    data: MNISTData, 
    batchIndex: Int, 
    batchSize: Int
  ): Seq[Tensor2[Height, Width]] = {
    val startIdx = batchIndex * batchSize
    val endIdx = math.min(startIdx + batchSize, data.images.length)
    data.images.slice(startIdx, endIdx).toSeq
  }
  
  /**
   * Get a batch of images as flattened tensors (for models expecting 1D input)
   */
  def getBatch[Feature <: Singleton](
    data: MNISTData, 
    batchIndex: Int, 
    batchSize: Int
  ): Seq[Tensor1[Feature]] = {
    val startIdx = batchIndex * batchSize
    imagesToTensors[Feature](data.images, startIdx, batchSize)
  }
  
  
  /**
   * Print ASCII representation of an MNIST digit for debugging
   */
  def printDigit(image: Tensor2[Height, Width], label: Option[Int] = None): Unit = {
    label.foreach(l => println(s"Label: $l"))
    println("ASCII representation:")
    
    // Convert JAX tensor to Python array for easier access
    val pythonArray = Jax.jnp.array(image.jaxValue).tolist()
    val scalaArray = pythonArray.as[Seq[Seq[Float]]]
    
    for (row <- 0 until 28) {
      val line = (0 until 28).map { col =>
        val pixel = scalaArray(row)(col)
        if (pixel < 0.1f) " "
        else if (pixel < 0.3f) "."
        else if (pixel < 0.6f) "o"
        else "#"
      }.mkString
      println(line)
    }
  }
}
