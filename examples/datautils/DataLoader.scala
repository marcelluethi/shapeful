package examples.datautils

import scala.language.experimental.namedTypeArguments
import shapeful.random.Random
import shapeful.*
import scala.collection.immutable.LazyList
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration.*

/** A functional data loader with efficient batching */
trait DataLoader[+Input, +Target]:
  def size: Int
  def apply(index: Int): (Input, Target)
  
  /** Get a batch of samples - this is the key method for efficiency */
  def getBatch(indices: Seq[Int]): (Vector[Input], Vector[Target])
  
  /** Convenience methods built on top of getBatch */
  def batch(indices: Seq[Int]): (LazyList[Input], LazyList[Target]) =
    val (inputs, targets) = getBatch(indices)
    (inputs.to(LazyList), targets.to(LazyList))

  def batch(start: Int, batchSize: Int): (LazyList[Input], LazyList[Target]) =
    batch(start until (start + batchSize).min(size))

  def iterator: Iterator[(Input, Target)] =
    (0 until size).iterator.map(apply)

  def batches(batchSize: Int): Iterator[(LazyList[Input], LazyList[Target])] =
    (0 until size by batchSize).iterator.map(start => batch(start, batchSize))

  /** Transform inputs while keeping targets */
  def mapInput[NewInput](f: Input => NewInput): DataLoader[NewInput, Target] =
    DataLoader.Mapped(this, f, identity)

  /** Transform targets while keeping inputs */
  def mapTarget[NewTarget](f: Target => NewTarget): DataLoader[Input, NewTarget] =
    DataLoader.Mapped(this, identity, f)

  /** Transform both inputs and targets */
  def map[NewInput, NewTarget](
      inputF: Input => NewInput,
      targetF: Target => NewTarget
  ): DataLoader[NewInput, NewTarget] =
    DataLoader.Mapped(this, inputF, targetF)

  /** Create a shuffled view of this data loader */
  def shuffle(key: Random.Key): DataLoader[Input, Target] =
    val indices = Tensor1["sample"](0 until size map (_.toFloat))
    val shuffledIndices = Random.permutation[PermutationAxis = "sample"](indices, key)
    val shuffledSeq = (0 until size).map(i => shuffledIndices.jaxValue.item(i).as[Float].toInt)
    DataLoader.Subset(this, shuffledSeq)

  /** Take only the first n samples */
  def take(n: Int): DataLoader[Input, Target] =
    DataLoader.Subset(this, 0 until n.min(size))

  /** Drop the first n samples */
  def drop(n: Int): DataLoader[Input, Target] =
    DataLoader.Subset(this, n until size)

  /** Create a subset using the given indices */
  def subset(indices: Seq[Int]): DataLoader[Input, Target] =
    DataLoader.Subset(this, indices)

object DataLoader:

  /** Create a data loader from sequences */
  def apply[Input, Target](inputs: Seq[Input], targets: Seq[Target]): DataLoader[Input, Target] =
    require(inputs.size == targets.size, s"Inputs (${inputs.size}) and targets (${targets.size}) must have same size")
    FromSeqs(inputs, targets)

  /** Create a data loader from a sequence of pairs */
  def fromPairs[Input, Target](data: Seq[(Input, Target)]): DataLoader[Input, Target] =
    FromSeqs(data.map(_._1), data.map(_._2))

  /** Create a data loader that generates data on-demand */
  def fromGenerator[Input, Target](
      dataSize: Int
  )(generator: Int => (Input, Target)): DataLoader[Input, Target] =
    Generated(dataSize, generator)

  // Efficient implementations as case classes

  private case class FromSeqs[Input, Target](inputs: Seq[Input], targets: Seq[Target]) extends DataLoader[Input, Target]:
    def size: Int = inputs.size
    def apply(index: Int): (Input, Target) = (inputs(index), targets(index))
    
    override def getBatch(indices: Seq[Int]): (Vector[Input], Vector[Target]) =
      val inputBatch = indices.map(inputs(_)).toVector
      val targetBatch = indices.map(targets(_)).toVector
      (inputBatch, targetBatch)

  private case class Generated[Input, Target](
    dataSize: Int, 
    generator: Int => (Input, Target)
  ) extends DataLoader[Input, Target]:
    def size: Int = dataSize
    def apply(index: Int): (Input, Target) =
      require(index >= 0 && index < size, s"Index $index out of bounds [0, $size)")
      generator(index)
    
    override def getBatch(indices: Seq[Int]): (Vector[Input], Vector[Target]) =
      val batch = indices.map(generator).toVector
      (batch.map(_._1), batch.map(_._2))

  private case class Mapped[Input, Target, NewInput, NewTarget](
    underlying: DataLoader[Input, Target],
    inputF: Input => NewInput,
    targetF: Target => NewTarget
  ) extends DataLoader[NewInput, NewTarget]:
    def size: Int = underlying.size
    def apply(index: Int): (NewInput, NewTarget) =
      val (input, target) = underlying(index)
      (inputF(input), targetF(target))
    
    override def getBatch(indices: Seq[Int]): (Vector[NewInput], Vector[NewTarget]) =
      val (inputs, targets) = underlying.getBatch(indices)
      (inputs.map(inputF), targets.map(targetF))

  private case class Subset[Input, Target](
    underlying: DataLoader[Input, Target],
    indices: Seq[Int]
  ) extends DataLoader[Input, Target]:
    def size: Int = indices.size
    def apply(index: Int): (Input, Target) =
      require(index >= 0 && index < size, s"Index $index out of bounds [0, $size)")
      underlying(indices(index))
    
    override def getBatch(batchIndices: Seq[Int]): (Vector[Input], Vector[Target]) =
      val originalIndices = batchIndices.map(indices(_))
      underlying.getBatch(originalIndices)

/** Utility functions for working with data loaders */
object DataLoaderOps:

  /** Split into training and validation sets */
  def split[Input, Target](
      loader: DataLoader[Input, Target],
      trainRatio: Double,
      key: Random.Key
  ): (DataLoader[Input, Target], DataLoader[Input, Target]) =
    require(trainRatio > 0 && trainRatio < 1, "Train ratio must be between 0 and 1")

    val shuffled = loader.shuffle(key)
    val trainSize = (loader.size * trainRatio).toInt
    (shuffled.take(trainSize), shuffled.drop(trainSize))

  /** Convert to materialized sequences (loads all data into memory) */
  def toSeqs[Input, Target](loader: DataLoader[Input, Target]): (Seq[Input], Seq[Target]) =
    val samples = loader.iterator.toSeq
    (samples.map(_._1), samples.map(_._2))

