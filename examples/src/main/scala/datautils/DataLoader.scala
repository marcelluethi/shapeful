package examples.datautils

import shapeful.random.Random
import shapeful.*
import scala.collection.immutable.{LazyList, ArraySeq}
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration.*

/** A functional data loader with efficient batching */
trait DataLoader[Features <: Tuple, Target <: Tuple]:
  def size: Int
  def apply(index: Int): (Tensor[Features], Tensor[Target])

  /** Get a batch of samples - this is the key method for efficiency */
  def getBatch[Sample <: Label](
      batchIndex: Int,
      batchSize: Int
  ): (Tensor[Sample *: Features], Tensor[Sample *: Target]) =
    val startIdx = batchIndex * batchSize
    val endIdx = math.min(startIdx + batchSize, size)
    val indices = (startIdx until endIdx).toSeq

    if indices.isEmpty then
      // Return empty tensors with correct shape
      throw new IndexOutOfBoundsException(s"Batch index $batchIndex out of bounds")

    getBatchByIndices[Sample](indices)

  /** Get a batch by specific indices */
  def getBatchByIndices[Sample <: Label](indices: Seq[Int]): (Tensor[Sample *: Features], Tensor[Sample *: Target]) =
    require(indices.nonEmpty, "Indices cannot be empty")

    // Get individual samples
    val samples = indices.map(apply)
    val (features, targets) = samples.unzip

    // Stack into batched tensors
    val batchedFeatures = Tensor.stack(Axis[Sample], features)
    val batchedTargets = Tensor.stack(Axis[Sample], targets)

    (batchedFeatures, batchedTargets)

  def batches[Sample <: Label](batchSize: Int): Iterator[(Tensor[Sample *: Features], Tensor[Sample *: Target])] =
    val numBatches = (size + batchSize - 1) / batchSize // Ceiling division
    (0 until numBatches).iterator.map { batchIndex =>
      getBatch[Sample](batchIndex, batchSize)
    }

  /** Create a shuffled view of this data loader */
  def shuffle(key: Random.Key): DataLoader[Features, Target] =
    new ShuffledDataLoader(this, key)

  /** Take only the first n samples */
  def take(n: Int): DataLoader[Features, Target] =
    new SlicedDataLoader(this, 0, math.min(n, size))

  /** Drop the first n samples */
  def drop(n: Int): DataLoader[Features, Target] =
    new SlicedDataLoader(this, n, size)

// Implementation classes for data loader operations
private class ShuffledDataLoader[Features <: Tuple, Target <: Tuple](
    underlying: DataLoader[Features, Target],
    key: Random.Key
) extends DataLoader[Features, Target]:
  private val shuffledIndices: Array[Int] =
    // Create a tensor of indices and shuffle using Random.permutation
    type IndexLabel = "Index"
    val indices = Tensor1.fromInts[IndexLabel]((0 until underlying.size).toSeq, DType.Int32)
    val shuffled = Random.permutation(Axis[IndexLabel], key, indices)
    // Convert back to Array[Int]
    (0 until underlying.size).map { i =>
      shuffled.at(Tuple1(i)).get.jaxValue.item().as[Int]
    }.toArray

  def size: Int = underlying.size

  def apply(index: Int): (Tensor[Features], Tensor[Target]) =
    underlying(shuffledIndices(index))

  override def getBatchByIndices[Sample <: Label](
      indices: Seq[Int]
  ): (Tensor[Sample *: Features], Tensor[Sample *: Target]) =
    val mappedIndices = indices.map(shuffledIndices(_))
    underlying.getBatchByIndices[Sample](mappedIndices)

private class SlicedDataLoader[Features <: Tuple, Target <: Tuple](
    underlying: DataLoader[Features, Target],
    start: Int,
    end: Int
) extends DataLoader[Features, Target]:
  require(start >= 0 && start <= underlying.size, s"Start index $start out of bounds")
  require(end >= start && end <= underlying.size, s"End index $end out of bounds")

  def size: Int = end - start

  def apply(index: Int): (Tensor[Features], Tensor[Target]) =
    require(index >= 0 && index < size, s"Index $index out of bounds for sliced loader")
    underlying(start + index)

  override def getBatchByIndices[Sample <: Label](
      indices: Seq[Int]
  ): (Tensor[Sample *: Features], Tensor[Sample *: Target]) =
    val mappedIndices = indices.map(_ + start)
    underlying.getBatchByIndices[Sample](mappedIndices)

/** Utility functions for working with data loaders */
object DataLoaderOps:

  /** Split into training and validation sets */
  def split[Features <: Tuple, Target <: Tuple](
      loader: DataLoader[Features, Target],
      trainRatio: Double
  ): (DataLoader[Features, Target], DataLoader[Features, Target]) =
    require(trainRatio > 0 && trainRatio < 1, "Train ratio must be between 0 and 1")

    val trainSize = (loader.size * trainRatio).toInt
    val trainLoader = loader.take(trainSize)
    val valLoader = loader.drop(trainSize)

    (trainLoader, valLoader)
