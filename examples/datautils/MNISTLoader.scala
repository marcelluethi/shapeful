package examples.datautils

import shapeful.*
import shapeful.tensor.Device
import shapeful.tensor.TensorIndexing.*
import shapeful.jax.Jax
import examples.datautils.DataLoader
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}
import scala.util.{Try, Success, Failure}
import me.shadaj.scalapy.py.SeqConverters
import scala.concurrent.{Future, ExecutionContext}
import java.util.concurrent.Executors
import scala.collection.immutable.ArraySeq
import examples.Utils

/** Simple MNIST data loader that loads all data into memory arrays.
  *
  * The MNIST dataset files use a specific binary format:
  *   - Images: magic number (2051), number of images, rows, cols, then pixel data
  *   - Labels: magic number (2049), number of labels, then label data
  */
object MNISTLoader:

  // Type labels for MNIST image dimensions
  type Height = "height"
  type Width = "width"
  type Sample = "Sample"
  type Label = "Label"

  /** MNIST dataset that loads all data into memory arrays
    */

  case class MNISTDataset(
      imagePixels: Tensor3[Sample, Height, Width], // Pre-loaded pixel data [imageIndex][flattenedPixels]
      labels: Tensor1[Label] // Pre-loaded labels
  ) extends DataLoader[(Height, Width), Tuple1[Label]]:

    // DataLoader trait implementation - only required methods
    def size: Int = imagePixels.shape.dim[Sample]

    def apply(index: Int): (Tensor[(Height, Width)], Tensor[Tuple1[Label]]) =
      (getImage(index), getLabelTensor(index))

    // PERFORMANCE OPTIMIZATION: Override getBatchByIndices to use vectorized JAX indexing
    // This performs 1 JAX operation instead of N individual slices, providing massive speedup!
    override def getBatchByIndices[BatchSample <: shapeful.Label](
        indices: Seq[Int]
    ): (Tensor[BatchSample *: (Height, Width)], Tensor[BatchSample *: Tuple1[Label]]) =
      require(indices.nonEmpty, "Indices cannot be empty")

      // Use JAX advanced indexing to get all images and labels at once
      val indexArray = Jax.jnp.array(indices.toPythonProxy)
      val batchedImagesJax = Jax.jnp.take(imagePixels.jaxValue, indexArray, axis = 0)
      val batchedLabelsJax = Jax.jnp.take(labels.jaxValue, indexArray, axis = 0)

      // Build shape for batched images: (batchSize, height, width)
      val imageShape = Shape[BatchSample *: (Height, Width)](
        shapeful.tensor.TupleHelpers.createTupleFromSeq(
          Seq(indices.size, imagePixels.shape.dim[Height], imagePixels.shape.dim[Width])
        )
      )

      // Build shape for batched labels: (batchSize, 1)
      val labelShape = Shape[BatchSample *: Tuple1[Label]](
        shapeful.tensor.TupleHelpers.createTupleFromSeq(Seq(indices.size, 1))
      )

      // Reshape labels to ensure they have the right shape (batchSize, 1)
      val reshapedLabelsJax = Jax.jnp.reshape(batchedLabelsJax, Seq(indices.size, 1).toPythonProxy)

      (
        new Tensor(imageShape, batchedImagesJax, DType.Float32),
        new Tensor(labelShape, reshapedLabelsJax, DType.Int32)
      )

    private def getLabelTensor(index: Int): Tensor[Tuple1[Label]] =
      if index < 0 || index >= size then
        throw new IndexOutOfBoundsException(s"Label index $index out of bounds [0, $size)")

      // Extract single label and keep as integer
      val labelTensor = labels.at(Tuple1(index)).get
      val labelValue = labelTensor.jaxValue.item().as[Int]
      Tensor1.fromInts[Label](ArraySeq(labelValue), DType.Int32)

    /** Extract a 2D image tensor from the 3D tensor
      */
    private def getImage(index: Int): Tensor[(Height, Width)] =
      if index < 0 || index >= size then
        throw new IndexOutOfBoundsException(s"Image index $index out of bounds [0, $size)")
      else
        // Extract the image at the given index from the Tensor3
        // Slice to get a 1x28x28 tensor, then remove the first dimension
        val sliced = imagePixels.slice[Sample](index, index + 1)
        // Remove the first dimension by reshaping to 2D
        val shape2D = Shape2[Height, Width](imagePixels.shape.dim[Height], imagePixels.shape.dim[Width])
        sliced.reshape(shape2D)

  /** Read 32-bit big-endian integer from DataInputStream
    */
  private def readInt(dis: DataInputStream): Int =
    dis.readInt()

  /** Load MNIST images from IDX3 format file into memory as Tensor3 Format: magic(4), numImages(4), rows(4), cols(4),
    * pixels...
    * @param filename
    *   Path to the MNIST image file
    * @param maxImages
    *   Maximum number of images to load (None means load all)
    */
  private def loadImagePixels(filename: String, maxImages: Option[Int] = None): Try[Tensor3[Sample, Height, Width]] =
    Try {
      val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
      try
        val magic = readInt(dis)
        if magic != 2051 then
          throw new IllegalArgumentException(s"Invalid magic number for images: $magic (expected 2051)")

        val totalImages = readInt(dis)
        val rows = readInt(dis)
        val cols = readInt(dis)

        val numImages = maxImages.map(max => math.min(max, totalImages)).getOrElse(totalImages)
        println(s"Loading $numImages of $totalImages images (${rows}x${cols}) from $filename into memory as Tensor3")

        // Read all pixel data at once
        val totalPixels = numImages * rows * cols
        val pixelBytes = new Array[Byte](totalPixels)
        dis.readFully(pixelBytes)
        // Convert bytes to floats with vectorized operation
        val allPixels = pixelBytes.map(b => (b & 0xff) / 255.0f)
        val shape = Shape3[Sample, Height, Width](numImages, rows, cols)
        val tensor = Tensor3.fromArray[Sample, Height, Width](shape, ArraySeq.unsafeWrapArray(allPixels), DType.Float32)
        tensor.toDevice(Device.CPU)
      finally dis.close()
    }

  /** Load MNIST labels from IDX1 format file into memory as Tensor1 Format: magic(4), numLabels(4), labels...
    * @param filename
    *   Path to the MNIST labels file
    * @param maxLabels
    *   Maximum number of labels to load (None means load all)
    */
  private def loadLabelsArray(filename: String, maxLabels: Option[Int] = None): Try[Tensor1[Label]] = Try {
    val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    try
      val magic = readInt(dis)
      if magic != 2049 then
        throw new IllegalArgumentException(s"Invalid magic number for labels: $magic (expected 2049)")

      val totalLabels = readInt(dis)
      val numLabels = maxLabels.map(max => math.min(max, totalLabels)).getOrElse(totalLabels)
      println(s"Loading $numLabels of $totalLabels labels from $filename into memory as Tensor1")

      val labels = Array.ofDim[Int](numLabels)
      for i <- 0.until(numLabels) do labels(i) = dis.readUnsignedByte()

      // Create Tensor1 from labels - specify the label type correctly
      val tensor = Tensor1.fromInts[Label](ArraySeq.unsafeWrapArray(labels), DType.Int32)
      tensor.toDevice(Device.CPU)
    finally dis.close()
  }

  /** Create MNIST dataset by loading all data into memory
    * @param imagesFile
    *   Path to the MNIST images file
    * @param labelsFile
    *   Path to the MNIST labels file
    * @param maxSamples
    *   Maximum number of samples to load (None means load all)
    */
  def createDataset(imagesFile: String, labelsFile: String, maxSamples: Option[Int] = None): Try[MNISTDataset] =
    for
      imagePixels <- loadImagePixels(imagesFile, maxSamples)
      labels <- loadLabelsArray(labelsFile, maxSamples)
    yield
      val numImages = imagePixels.shape.dim[Sample]
      val numLabels = labels.shape.size
      if numImages != numLabels then
        throw new IllegalArgumentException(s"Mismatch: $numImages images vs $numLabels labels")
      println(s"Created in-memory MNIST dataset with $numImages images")
      MNISTDataset(imagePixels, labels)

  /** Create training dataset (loads all data into memory)
    * @param dataDir
    *   Directory containing the MNIST data files
    * @param maxSamples
    *   Maximum number of samples to load (None means load all 60000)
    */
  def createTrainingDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[MNISTDataset] =
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset(imagesFile, labelsFile, maxSamples)

  /** Create test dataset (loads all data into memory)
    * @param dataDir
    *   Directory containing the MNIST data files
    * @param maxSamples
    *   Maximum number of samples to load (None means load all 10000)
    */
  def createTestDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[MNISTDataset] =
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    createDataset(imagesFile, labelsFile, maxSamples)
