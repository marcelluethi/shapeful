package examples.datautils

import scala.language.experimental.namedTypeArguments
import shapeful.random.Random
import shapeful.*
import scala.collection.immutable.LazyList

trait DataLoader[+Input, +Target] {
  
  /** Total number of samples in the dataset */
  def size: Int
  
  /** Get a single sample by index */
  def apply(index: Int): (Input, Target)
  
  /** Get a batch of samples as lazy sequences */
  def batch(indices: Seq[Int]): (LazyList[Input], LazyList[Target]) = {
    val samples = LazyList.from(indices).map(apply)
    (samples.map(_._1), samples.map(_._2))
  }
  
  /** Get a batch of samples by range */
  def batch(start: Int, size: Int): (LazyList[Input], LazyList[Target]) = 
    batch(start until (start + size).min(this.size))
  
  /** Lazy iterator over all samples */
  def iterator: Iterator[(Input, Target)] = 
    (0 until size).iterator.map(apply)
  
  /** Lazy iterator over batches of given size */
  def batches(batchSize: Int): Iterator[(LazyList[Input], LazyList[Target])] =
    (0 until size by batchSize).iterator.map(start => batch(start, batchSize))
  
  /** Transform inputs while keeping targets */
  def mapInput[NewInput](f: Input => NewInput): DataLoader[NewInput, Target] =
    DataLoader.mapped(this, f, identity)
  
  /** Transform targets while keeping inputs */
  def mapTarget[NewTarget](f: Target => NewTarget): DataLoader[Input, NewTarget] =
    DataLoader.mapped(this, identity, f)
  
  /** Transform both inputs and targets */
  def map[NewInput, NewTarget](
    inputF: Input => NewInput, 
    targetF: Target => NewTarget
  ): DataLoader[NewInput, NewTarget] =
    DataLoader.mapped(this, inputF, targetF)
  
    /** Create a shuffled view of this data loader */
  def shuffle(key: Random.Key): DataLoader[Input, Target] =
    DataLoader.shuffled(this, key)
  
  /** Take only the first n samples */
  def take(n: Int): DataLoader[Input, Target] =
    DataLoader.subset(this, 0 until n.min(size))
  
  /** Drop the first n samples */
  def drop(n: Int): DataLoader[Input, Target] =
    DataLoader.subset(this, n until size)
  
  /** Create a subset using the given indices */
  def subset(indices: Seq[Int]): DataLoader[Input, Target] =
    DataLoader.subset(this, indices)
}


object DataLoader {
  
  /** Create a data loader from sequences */
  def apply[Input, Target](inputs: Seq[Input], targets: Seq[Target]): DataLoader[Input, Target] = {
    require(inputs.size == targets.size, s"Inputs (${inputs.size}) and targets (${targets.size}) must have same size")
    fromSeqs(inputs, targets)
  }
  
  /** Create a data loader from a sequence of pairs */
  def fromPairs[Input, Target](data: Seq[(Input, Target)]): DataLoader[Input, Target] =
    fromSeqs(data.map(_._1), data.map(_._2))
  
  /** Create a data loader that generates data on-demand */
  def fromGenerator[Input, Target](
    dataSize: Int
  )(generator: Int => (Input, Target)): DataLoader[Input, Target] = 
    new DataLoader[Input, Target] {
      def size: Int = dataSize
      def apply(index: Int): (Input, Target) = {
        require(index >= 0 && index < size, s"Index $index out of bounds [0, $size)")
        generator(index)
      }
    }
  
  // Internal implementations
  
  private def fromSeqs[Input, Target](inputs: Seq[Input], targets: Seq[Target]): DataLoader[Input, Target] =
    new DataLoader[Input, Target] {
      def size: Int = inputs.size
      def apply(index: Int): (Input, Target) = (inputs(index), targets(index))
    }
  
  private def mapped[Input, Target, NewInput, NewTarget](
    underlying: DataLoader[Input, Target],
    inputF: Input => NewInput,
    targetF: Target => NewTarget
  ): DataLoader[NewInput, NewTarget] = 
    new DataLoader[NewInput, NewTarget] {
      def size: Int = underlying.size
      def apply(index: Int): (NewInput, NewTarget) = {
        val (input, target) = underlying(index)
        (inputF(input), targetF(target))
      }
    }
  
  private def shuffled[Input, Target](
    underlying: DataLoader[Input, Target], 
    key: Random.Key
  ): DataLoader[Input, Target] = {
    // Create a tensor with indices 0, 1, 2, ..., size-1
    val indices = Tensor1["sample"](0 until underlying.size map (_.toFloat))
    
    // Use Random.permutation to shuffle the indices
    val shuffledIndices = Random.permutation[PermutationAxis="sample"](indices, key)
    
    // Extract the shuffled indices as a sequence
    val shuffledSeq = (0 until underlying.size).map(i => 
      shuffledIndices.jaxValue.item(i).as[Float].toInt
    )
    
    subset(underlying, shuffledSeq)
  }
  
  private def subset[Input, Target](
    underlying: DataLoader[Input, Target],
    indices: Seq[Int]
  ): DataLoader[Input, Target] =
    new DataLoader[Input, Target] {
      def size: Int = indices.size
      def apply(index: Int): (Input, Target) = {
        require(index >= 0 && index < size, s"Index $index out of bounds [0, $size)")
        underlying(indices(index))
      }
    }
}

/** Utility functions for working with data loaders */
object DataLoaderOps {
  
  /** Split into training and validation sets */
  def split[Input, Target](loader: DataLoader[Input, Target], trainRatio: Double, key: Random.Key): (DataLoader[Input, Target], DataLoader[Input, Target]) = {
    require(trainRatio > 0 && trainRatio < 1, "Train ratio must be between 0 and 1")
    
    val shuffled = loader.shuffle(key)
    val trainSize = (loader.size * trainRatio).toInt
    (shuffled.take(trainSize), shuffled.drop(trainSize))
  }
  
  /** Convert to materialized sequences (loads all data into memory) */
  def toSeqs[Input, Target](loader: DataLoader[Input, Target]): (Seq[Input], Seq[Target]) = {
    val samples = loader.iterator.toSeq
    (samples.map(_._1), samples.map(_._2))
  }
}