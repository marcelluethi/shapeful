package examples.datautils

import scala.language.experimental.namedTypeArguments
import scala.annotation.experimental
import shapeful.*
import shapeful.jax.Jax
import examples.datautils.DataLoader
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}
import scala.util.{Try, Success, Failure}
import me.shadaj.scalapy.py.SeqConverters
import scala.concurrent.{Future, ExecutionContext}
import java.util.concurrent.Executors

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

  /** MNIST dataset that loads all data into memory arrays
    */
  @experimental
  case class MNISTDataset(
      imagePixels: Array[Array[Array[Float]]], // Pre-loaded pixel data [imageIndex][row][col]
      labels: Array[Int], // Pre-loaded labels
      rows: Int = 28,
      cols: Int = 28
  ) extends DataLoader[Tensor2[Height, Width], Int]:

    // DataLoader trait implementation - only required methods
    def size: Int = imagePixels.length

    def apply(index: Int): (Tensor2[Height, Width], Int) = 
      (getImage(index), getLabel(index))

    def getBatch(indices: Seq[Int]): (Vector[Tensor2[Height, Width]], Vector[Int]) =
      val images = indices.map(getImage).toVector
      val batchLabels = indices.map(getLabel).toVector
      (images, batchLabels)

    /** Convert pre-loaded pixel data to 2D Tensor
      */
    private def getImage(index: Int): Tensor2[Height, Width] =
      if index < 0 || index >= size then
        throw new IndexOutOfBoundsException(s"Image index $index out of bounds [0, $size)")
      
      val pixelRows = imagePixels(index).map(_.toSeq).toSeq
      Tensor2[Height, Width](pixelRows, DType.Float32)

    /** Get pre-loaded label
      */
    private def getLabel(index: Int): Int =
      if index < 0 || index >= size then
        throw new IndexOutOfBoundsException(s"Label index $index out of bounds [0, $size)")
      
      labels(index)

  /** Read 32-bit big-endian integer from DataInputStream
    */
  private def readInt(dis: DataInputStream): Int =
    dis.readInt()

  /** Load MNIST images from IDX3 format file into memory arrays
    * Format: magic(4), numImages(4), rows(4), cols(4), pixels...
    */
  private def loadImagePixels(filename: String): Try[Array[Array[Array[Float]]]] = Try {
    val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    try
      val magic = readInt(dis)
      if magic != 2051 then
        throw new IllegalArgumentException(s"Invalid magic number for images: $magic (expected 2051)")

      val numImages = readInt(dis)
      val rows = readInt(dis)
      val cols = readInt(dis)

      println(s"Loading $numImages images of size ${rows}x${cols} from $filename into memory")

      val imagePixels = Array.ofDim[Array[Array[Float]]](numImages)

      for i <- 0 until numImages do
        val pixelRows = Array.ofDim[Array[Float]](rows)
        for row <- 0 until rows do
          val rowPixels = Array.ofDim[Float](cols)
          for col <- 0 until cols do
            // Read unsigned byte and normalize to [0, 1]
            val pixel = dis.readUnsignedByte()
            rowPixels(col) = pixel / 255.0f
          pixelRows(row) = rowPixels
        imagePixels(i) = pixelRows

      imagePixels
    finally dis.close()
  }

  /** Load MNIST labels from IDX1 format file into memory array
    * Format: magic(4), numLabels(4), labels...
    */
  private def loadLabelsArray(filename: String): Try[Array[Int]] = Try {
    val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
    try
      val magic = readInt(dis)
      if magic != 2049 then
        throw new IllegalArgumentException(s"Invalid magic number for labels: $magic (expected 2049)")

      val numLabels = readInt(dis)
      println(s"Loading $numLabels labels from $filename into memory")

      val labels = Array.ofDim[Int](numLabels)
      for i <- 0 until numLabels do labels(i) = dis.readUnsignedByte()

      labels
    finally dis.close()
  }

  /** Create MNIST dataset by loading all data into memory
    */
  def createDataset(imagesFile: String, labelsFile: String): Try[MNISTDataset] =
    for
      imagePixels <- loadImagePixels(imagesFile)
      labels <- loadLabelsArray(labelsFile)
    yield
      if imagePixels.length != labels.length then
        throw new IllegalArgumentException(s"Mismatch: ${imagePixels.length} images vs ${labels.length} labels")
      println(s"Created in-memory MNIST dataset with ${imagePixels.length} images")
      MNISTDataset(imagePixels, labels)

  /** Create training dataset (loads all data into memory)
    */
  def createTrainingDataset(dataDir: String = "data"): Try[MNISTDataset] =
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset(imagesFile, labelsFile)

  /** Create test dataset (loads all data into memory)
    */
  def createTestDataset(dataDir: String = "data"): Try[MNISTDataset] =
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    createDataset(imagesFile, labelsFile)
