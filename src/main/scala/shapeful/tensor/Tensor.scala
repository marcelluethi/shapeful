package shapeful.tensor

import scala.annotation.targetName
import scala.collection.immutable.ArraySeq
import shapeful.jax.Jax
import shapeful.jax.JaxDType
import shapeful.Label
import shapeful.tensor.TupleHelpers.ToIntTuple
import shapeful.random.Random
import me.shadaj.scalapy.py.SeqConverters
import shapeful.jax.Jax.PyDynamic

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])

class Tensor[T <: Tuple](val shape: Shape[T], val jaxValue: Jax.PyDynamic, val dtype: DType = DType.Float32):

  def toDevice(device: Device): Tensor[T] =
    val newJaxValue = Jax.device_put(jaxValue, device.jaxDevice)
    new Tensor(shape, newJaxValue, dtype)

  /** map a function over the given axis of the tensor
    */
  inline def vmap[VmapAxis <: Label, OuterShape <: Tuple](
      axis: Axis[VmapAxis],
      f: Tensor[TupleHelpers.Remove[VmapAxis, T]] => Tensor[OuterShape]
  ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =

    // The axes that our mapping functions maps over
    type InnerAxes = TupleHelpers.Remove[VmapAxis, T]
    val innerAxesInd = TupleHelpers.indicesOf[InnerAxes, T]

    // the index of the axis over which we map
    val vmapAxisIndex = TupleHelpers.indexOf[VmapAxis, T]

    val allDims = shape.dims
    val innerDims = allDims.zipWithIndex.filter(_._2 != vmapAxisIndex).map(_._1)
    val innerShapeTuple = TupleHelpers.createTupleFromSeq[InnerAxes](innerDims)

    type ResultTuple = Tuple.Concat[Tuple1[VmapAxis], OuterShape]
    val resultTupleAxesInd = TupleHelpers.indicesOf[InnerAxes, T]

    // Create the function for JAX vmap
    val fpy = (jxpr: Jax.PyDynamic) =>
      // Create inner tensor with correct dimensions
      val innerShape = Shape[InnerAxes](
        innerShapeTuple
      )
      val innerTensor = new Tensor[InnerAxes](innerShape, jxpr, dtype)

      // Apply user function
      val result = f(innerTensor)
      result.jaxValue

    // call the actual vmap function
    val vmap_val = Jax.jax_helper.vmap(fpy, vmapAxisIndex)(jaxValue)

    // Get result shape from JAX output
    val resultDims = vmap_val.shape.as[Seq[Int]]
    val resultTuple = TupleHelpers.createTupleFromSeq[ResultTuple](resultDims)

    // ✅ Create shape with correct type annotation
    val resultShape = Shape[ResultTuple](resultTuple)

    new Tensor(resultShape, vmap_val, dtype)

  /** zip two tensors and apply a function over the given axis of both tensors.
    */
  inline def zipVmap[VmapAxis <: Label, OtherShape <: Tuple, OuterShape <: Tuple](
      axis: Axis[VmapAxis],
      other: Tensor[OtherShape]
  )(
      f: (
          Tensor[TupleHelpers.Remove[VmapAxis, T]],
          Tensor[TupleHelpers.Remove[VmapAxis, OtherShape]]
      ) => Tensor[OuterShape]
  ): Tensor[Tuple.Concat[Tuple1[VmapAxis], OuterShape]] =

    // The axes that our mapping functions maps over
    type InnerAxes1 = TupleHelpers.Remove[VmapAxis, T]
    type InnerAxes2 = TupleHelpers.Remove[VmapAxis, OtherShape]

    val vmapAxisIndex1 = TupleHelpers.indexOf[VmapAxis, T]
    val vmapAxisIndex2 = TupleHelpers.indexOf[VmapAxis, OtherShape]

    val allDims1 = shape.dims
    val innerDims1 = allDims1.zipWithIndex.filter(_._2 != vmapAxisIndex1).map(_._1)
    val innerShapeTuple1 = TupleHelpers.createTupleFromSeq[InnerAxes1](innerDims1)

    val allDims2 = other.shape.dims
    val innerDims2 = allDims2.zipWithIndex.filter(_._2 != vmapAxisIndex2).map(_._1)
    val innerShapeTuple2 = TupleHelpers.createTupleFromSeq[InnerAxes2](innerDims2)

    type ResultTuple = Tuple.Concat[Tuple1[VmapAxis], OuterShape]

    // Create the function for JAX vmap
    val fpy = (jxpr1: Jax.PyDynamic, jxpr2: Jax.PyDynamic) =>
      // Create inner tensor with correct dimensions
      val innerShape1 = Shape[InnerAxes1](innerShapeTuple1)
      val innerShape2 = Shape[InnerAxes2](innerShapeTuple2)
      val innerTensor1 = new Tensor[InnerAxes1](innerShape1, jxpr1, dtype)
      val innerTensor2 = new Tensor[InnerAxes2](innerShape2, jxpr2, dtype)

      // Apply user function
      val result = f(innerTensor1, innerTensor2)
      result.jaxValue

    // call the actual vmap function
    val vmap_val = Jax.jax_helper.vmap2(fpy, (vmapAxisIndex1, vmapAxisIndex2))(jaxValue, other.jaxValue)

    // Get result shape from JAX output
    val resultDims = vmap_val.shape.as[Seq[Int]]
    val resultTuple = TupleHelpers.createTupleFromSeq[ResultTuple](resultDims)

    // ✅ Create shape with correct type annotation
    val resultShape = Shape[ResultTuple](resultTuple)

    new Tensor(resultShape, vmap_val, dtype)

  inline def unstack[VmapAxis <: Label](
      axis: Axis[VmapAxis]
  ): ArraySeq[Tensor[TupleHelpers.Remove[VmapAxis, T]]] =
    type InnerAxes = TupleHelpers.Remove[VmapAxis, T]
    val vmapAxisIndex = TupleHelpers.indexOf[VmapAxis, T]
    val vmapAxisSize = shape.dims(vmapAxisIndex)

    // Use JAX's split function to split along the axis
    val splitArrays = Jax.jnp.split(jaxValue, vmapAxisSize, axis = vmapAxisIndex)

    // Calculate inner shape (removing the split dimension)
    val allDims = shape.dims
    val innerDims = allDims.zipWithIndex.filter(_._2 != vmapAxisIndex).map(_._1)
    val innerShapeTuple = TupleHelpers.createTupleFromSeq[InnerAxes](innerDims)
    val innerShape = Shape[InnerAxes](innerShapeTuple)

    // Convert each split array to a tensor and squeeze out the split dimension
    (0 until vmapAxisSize)
      .map { i =>
        val splitArray = splitArrays.__getitem__(i)
        val squeezedArray = Jax.jnp.squeeze(splitArray, axis = vmapAxisIndex)
        new Tensor[InnerAxes](innerShape, squeezedArray, dtype)
      }
      .to(ArraySeq)

  def stack[NewAxis <: Label](
      axis: Axis[NewAxis],
      otherTensor: Tensor[T]*
  ): Tensor[Tuple.Concat[Tuple1[NewAxis], T]] =
    Tensor.stack(axis, (this +: otherTensor)*)

  inline def concat[ConcatAxis <: Label](
      axis: Axis[ConcatAxis],
      otherTensor: Tensor[T]*
  ): Tensor[T] =
    Tensor.concat(axis, (this +: otherTensor)*)

  /** Reshape the tensor to a new shape.
    */
  def reshape[NewT <: Tuple](newShape: Shape[NewT]): Tensor[NewT] =
    if shape.dims.product != newShape.dims.product then
      throw new IllegalArgumentException("New shape must have the same number of elements")
    val result = Jax.jnp.reshape(jaxValue, newShape.dims.toPythonProxy)
    new Tensor[NewT](newShape, result, dtype)

  /** change the labels of the tensor dimensions.
    */
  def relabel[Names <: Tuple](using ev: Tuple.Size[T] =:= Tuple.Size[Names]): Tensor[Names] =
    new Tensor[Names](shape.relabel[Names], jaxValue, dtype)

  /** Access a specific index of the tensor.
    */
  def at(idx: ToIntTuple[T]): TensorIndexer[T] =
    new TensorIndexer(this, idx)

  /** Change the dtype of the tensor.
    */
  def asType(newDType: DType): Tensor[T] =
    if dtype == newDType then
      // No conversion needed
      this
    else
      // Use JAX's astype function to convert dtype
      val convertedJaxValue = Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(newDType))
      new Tensor[T](shape, convertedJaxValue, newDType)

  /** Compare two tensors for equality.
    */
  def tensorEquals(other: Tensor[?]): Boolean =
    // First check if shapes match
    if this.shape.dims != other.shape.dims then false
    // Then check if dtypes match
    else if this.dtype != other.dtype then false
    else
      try
        // Use JAX's array_equal for deep value comparison
        val result = Jax.jnp.array_equal(this.jaxValue, other.jaxValue)
        result.item().as[Boolean]
      catch case _: Exception => false

  // Operator overload for ==
  def ==(other: Tensor[?]): Boolean = tensorEquals(other)

  // Operator overload for !=
  def !=(other: Tensor[?]): Boolean = !tensorEquals(other)

  /** Element-wise equality check between two tensors. Returns a new tensor of boolean values indicating equality.
    */
  def elementEquals[U <: Tuple](other: Tensor[U]): Tensor[T] =
    require(this.shape.dims == other.shape.dims, s"Shape mismatch: ${this.shape.dims} vs ${other.shape.dims}")

    val resultJax = Jax.jnp.equal(this.jaxValue, other.jaxValue)
    new Tensor[T](this.shape, resultJax, DType.Bool)

  /** Approximate equality check between two tensors.
    */
  def approxEquals(other: Tensor[?], tolerance: Float = 1e-6f): Boolean =
    if this.shape.dims != other.shape.dims then false
    else if this.dtype != other.dtype then false
    else
      try
        val result = Jax.jnp.allclose(
          this.jaxValue,
          other.jaxValue,
          atol = tolerance,
          rtol = tolerance
        )
        result.item().as[Boolean]
      catch case _: Exception => false

  // Override Object.equals for proper Scala equality semantics
  override def equals(obj: Any): Boolean = obj match
    case other: Tensor[?] => this.tensorEquals(other)
    case _                => false

  // Override hashCode to be consistent with equals
  override def hashCode(): Int =
    try
      val shapeHash = shape.dims.hashCode()
      val dtypeHash = dtype.hashCode()

      // For consistent hashing, we need to be deterministic about values
      val valueHash =
        if shape.dims.product <= 10 then
          try
            if shape.dims.isEmpty then
              // For scalars, use the actual value
              jaxValue.item().hashCode()
            else
              // For small tensors, flatten and convert to python list for consistent hashing
              val flattened = Jax.jnp.flatten(jaxValue)
              val pythonList = flattened.tolist()
              pythonList.hashCode()
          catch
            case _ =>
              // Fallback to shape-based hash
              shape.dims.product.hashCode()
        else
          // For large tensors, use shape and dtype only for performance
          (shape.dims.product, shape.dims.sum).hashCode()

      (shapeHash, dtypeHash, valueHash).hashCode()
    catch case _: Exception => (shape.dims, dtype).hashCode()

  override def toString: String =
    try
      val jaxArray = jaxValue.block_until_ready()

      // For scalar tensors (0D), use item() to get the scalar value
      shape.dims.size match
        case 0 => jaxArray.item().toString
        case _ => jaxArray.toString() // Let JAX handle the formatting for all dimensions
    catch
      case _: Exception =>
        s"Tensor[shape=${shape.dims.mkString("(", ", ", ")")}](dtype=${dtype.name})"

  def stats(): String =
    try
      val jaxValues = jaxValue.block_until_ready()
      val mean = Jax.jnp.mean(jaxValues).item().as[Float]
      val stdDev = Jax.jnp.std(jaxValues).item().as[Float]
      val min = Jax.jnp.min(jaxValues).item().as[Float]
      val max = Jax.jnp.max(jaxValues).item().as[Float]

      s"  Mean: $mean\t" +
        s"  StdDev: $stdDev\t" +
        s"  Min: $min\t" +
        s"  Max: $max"
    catch
      case ex: Exception =>
        s"Error calculating stats: ${ex.getMessage}"

  def inspect: String =
    try
      val jaxValues = jaxValue.block_until_ready()
      val pythonStr = jaxValues.toString()

      val shapeStr = shape.dims.mkString("(", ", ", ")")
      // Show shape info instead of trying to extract type names
      val infoStr = s"shape=${shapeStr}"

      s"Tensor[${infoStr}](\n" +
        s"  dtype: ${dtype.name}\n" +
        s"  values:\n${indentLines(pythonStr, "    ")}\n" +
        s")"
    catch
      case ex: Exception =>
        val shapeStr = shape.dims.mkString("(", ", ", ")")
        s"Tensor[shape=${shapeStr}](dtype: ${dtype.name}, values: <error: ${ex.getMessage}>)"

  // Helper method to indent each line of a multiline string
  private def indentLines(text: String, indent: String): String =
    text.split("\n").map(line => if line.trim.nonEmpty then indent + line else line).mkString("\n")

object Tensor:

  type Tensor0 = Tensor[EmptyTuple]
  type Tensor1[L <: Label] = Tensor[Tuple1[L]]
  type Tensor2[L1 <: Label, L2 <: Label] = Tensor[(L1, L2)]
  type Tensor3[L1 <: Label, L2 <: Label, L3 <: Label] = Tensor[(L1, L2, L3)]
  type Tensor4[L1 <: Label, L2 <: Label, L3 <: Label, L4 <: Label] = Tensor[(L1, L2, L3, L4)]

  def apply[T <: Tuple](shape: Shape[T], values: ArraySeq[Float], dtype: DType = DType.Float32): Tensor[T] =
    val jaxValues = Jax.jnp
      .array(
        values.toPythonProxy,
        dtype = JaxDType.jaxDtype(dtype)
      )
      .reshape(shape.dims.toPythonProxy)

    new Tensor[T](shape, jaxValues, dtype)

  // Backward compatibility method for Seq
  def apply[T <: Tuple](shape: Shape[T], values: Seq[Float]): Tensor[T] =
    apply(shape, ArraySeq.from(values), DType.Float32)

  def apply[T <: Tuple](shape: Shape[T], values: Seq[Float], dtype: DType): Tensor[T] =
    apply(shape, ArraySeq.from(values), dtype)

  def zeros[T <: Tuple](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    val jaxValues = Jax.jnp.zeros(shape.dims.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
    new Tensor[T](shape, jaxValues, dtype)

  def ones[T <: Tuple](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    val jaxValues = Jax.jnp.ones(shape.dims.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
    new Tensor[T](shape, jaxValues, dtype)

  /** Generate random tensor from standard normal distribution N(0, 1) */
  def randn[T <: Tuple](
      key: Random.Key,
      shape: Shape[T],
      dtype: DType = DType.Float32
  ): Tensor[T] =
    Random.normal(key, shape, dtype = dtype)

  /** Generate random tensor from uniform distribution [0, 1) */
  def rand[T <: Tuple](
      key: Random.Key,
      shape: Shape[T],
      dtype: DType = DType.Float32
  ): Tensor[T] =
    Random.uniform(key, shape, dtype)

  /** Generate random tensor from uniform distribution [minval, maxval) */
  def randUniform[T <: Tuple](
      key: Random.Key,
      shape: Shape[T],
      minval: Tensor0,
      maxval: Tensor0,
      dtype: DType = DType.Float32
  ): Tensor[T] =
    Random.uniform(key, shape, minval, maxval, dtype)

  /** stack a sequence of tensors along a new axis
    */
  def stack[T <: Tuple, NewAxis <: Label](
      axis: Axis[NewAxis],
      tensors: Tensor[T]*
  ): Tensor[Tuple.Concat[Tuple1[NewAxis], T]] =
    require(tensors.nonEmpty, "Cannot stack empty sequence of tensors")
    val refShape = tensors.head.shape
    val jaxValues = tensors.map(_.jaxValue).toPythonProxy
    val stacked = Jax.jnp.stack(jaxValues)
    val newShapeSeq = Seq(tensors.length) ++ refShape.dims
    val newShapeTuple = TupleHelpers.createTupleFromSeq[Tuple.Concat[Tuple1[NewAxis], T]](newShapeSeq)
    val shape = Shape[Tuple.Concat[Tuple1[NewAxis], T]](newShapeTuple)
    new Tensor(shape, stacked, tensors.head.dtype)

  /** Concat tensors along an existing axis
    */
  inline def concat[T <: Tuple, ConcatAxis <: Label](
      axis: Axis[ConcatAxis],
      tensors: Tensor[T]*
  ): Tensor[T] =
    require(tensors.nonEmpty, "Cannot concat empty sequence of tensors")
    val jaxValues = tensors.map(_.jaxValue).toPythonProxy
    val axisIndex = TupleHelpers.indexOf[ConcatAxis, T]
    val concatenated = Jax.jnp.concatenate(jaxValues, axis = axisIndex)

    // Get the actual shape directly from JAX result
    val resultDims = concatenated.shape.as[Seq[Int]]
    val newShapeTuple = TupleHelpers.createTupleFromSeq[T](resultDims)
    val shape = Shape[T](newShapeTuple)

    new Tensor(shape, concatenated, tensors.head.dtype)

object Tensor0:
  import Tensor.{Tensor0, Tensor1}

  def apply(value: Float | Int | Boolean): Tensor[EmptyTuple] =
    value match
      case v: Float   => new Tensor[EmptyTuple](Shape.empty, Jax.jnp.array(v), DType.Float32)
      case v: Int     => new Tensor[EmptyTuple](Shape.empty, Jax.jnp.array(v), DType.Int32)
      case v: Boolean => new Tensor[EmptyTuple](Shape.empty, Jax.jnp.array(v), DType.Bool)

object Tensor1:
  import Tensor.{Tensor1, Tensor2}

  def apply[L <: Label](values: ArraySeq[Float], dtype: DType = DType.Float32): Tensor[Tuple1[L]] =
    require(values.nonEmpty, "Cannot create tensor from empty sequence")
    val shape = Shape1[L](values.length)
    val jaxValues = Jax.jnp.array(values.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
    new Tensor(shape, jaxValues, dtype)

  // Backward compatibility method for Seq
  def apply[L <: Label](values: Seq[Float]): Tensor[Tuple1[L]] =
    apply(ArraySeq.from(values), DType.Float32)

  def apply[L <: Label](values: Seq[Float], dtype: DType): Tensor[Tuple1[L]] =
    apply(ArraySeq.from(values), dtype)

  def fromArray[L1 <: Label](
      values: ArraySeq[Float],
      dtype: DType = DType.Float32
  ): Tensor[Tuple1[L1]] =

    val shape = Shape1[L1](values.length)
    val jaxValues = Jax.jnp
      .array(values.toArray.toPythonCopy, dtype = JaxDType.jaxDtype(dtype))

    new Tensor(shape, jaxValues, dtype)

  def fromInts[L <: Label](values: ArraySeq[Int], dtype: DType = DType.Int32): Tensor[Tuple1[L]] =
    require(values.nonEmpty, "Cannot create tensor from empty sequence")
    val shape = Shape1[L](values.length)
    val jaxValues = Jax.jnp.array(values.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
    new Tensor(shape, jaxValues, dtype)

  // Backward compatibility method for Seq
  def fromInts[L <: Label](values: Seq[Int]): Tensor[Tuple1[L]] =
    fromInts(ArraySeq.from(values), DType.Int32)

  def fromInts[L <: Label](values: Seq[Int], dtype: DType): Tensor[Tuple1[L]] =
    fromInts(ArraySeq.from(values), dtype)

object Tensor2:

  import Tensor.{Tensor1, Tensor2, Tensor3}
  import Shape.{Shape1, Shape2, Shape3}

  def apply[L1 <: Label, L2 <: Label](
      values: ArraySeq[ArraySeq[Float]],
      dtype: DType = DType.Float32
  ): Tensor[(L1, L2)] =
    require(values.nonEmpty, "Cannot create tensor from empty sequence")
    require(values.forall(_.nonEmpty), "All rows must be non-empty")

    val rows = values.length
    val cols = values.head.length
    require(values.forall(_.length == cols), "All rows must have the same length")

    val shape = Shape2[L1, L2](rows, cols)
    val flatValues = values.flatten
    fromArray(shape, flatValues, dtype)

  def fromArray[L1 <: Label, L2 <: Label](
      shape: Shape2[L1, L2],
      values: ArraySeq[Float],
      dtype: DType = DType.Float32
  ): Tensor[(L1, L2)] =

    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val jaxValues = Jax.jnp
      .array(values.toArray.toPythonCopy, dtype = JaxDType.jaxDtype(dtype))
      .reshape(shape.dims(0), shape.dims(1))

    new Tensor(shape, jaxValues, dtype)

  // Backward compatibility method for Seq
  def apply[L1 <: Label, L2 <: Label](values: Seq[Seq[Float]]): Tensor[(L1, L2)] =
    apply(ArraySeq.from(values.map(ArraySeq.from)), DType.Float32)

  def apply[L1 <: Label, L2 <: Label](values: Seq[Seq[Float]], dtype: DType): Tensor[(L1, L2)] =
    apply(ArraySeq.from(values.map(ArraySeq.from)), dtype)

  def eye[L <: Label](shape: Shape1[L], dtype: DType = DType.Float32): Tensor2[L, L] =
    val jaxValues = Jax.jnp.eye(shape.size, dtype = JaxDType.jaxDtype(dtype))
    new Tensor2[L, L](Shape2(shape.size, shape.size), jaxValues, dtype)

  def fromDiag[L <: Label](diag: Tensor1[L], dtype: DType = DType.Float32): Tensor2[L, L] =
    val size = diag.shape.size
    val jaxValues = Jax.jnp.diag(diag.jaxValue)
    new Tensor2[L, L](Shape2(size, size), jaxValues, dtype)

object Tensor3:
  import Tensor.{Tensor1, Tensor2, Tensor3}
  import shapeful.tensor.Shape.{Shape1, Shape2, Shape3}

  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      values: ArraySeq[ArraySeq[ArraySeq[Float]]],
      dtype: DType = DType.Float32
  ): Tensor[(L1, L2, L3)] =
    require(values.nonEmpty, "Cannot create tensor from empty sequence")
    require(values.forall(_.nonEmpty), "All outer dimensions must be non-empty")
    require(values.forall(_.forall(_.nonEmpty)), "All inner dimensions must be non-empty")

    val dim1 = values.length
    val dim2 = values.head.length
    val dim3 = values.head.head.length

    require(values.forall(_.length == dim2), "All second dimensions must match")
    require(values.forall(_.forall(_.length == dim3)), "All third dimensions must match")

    val shape = Shape3[L1, L2, L3](dim1, dim2, dim3)
    val flatValues = values.flatten.flatten
    val jaxValues = Jax.jnp
      .array(flatValues.toPythonProxy, dtype = JaxDType.jaxDtype(dtype))
      .reshape(dim1, dim2, dim3)

    new Tensor(shape, jaxValues, dtype)

  // Backward compatibility method for Seq
  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      values: Seq[Seq[Seq[Float]]]
  ): Tensor[(L1, L2, L3)] =
    apply(ArraySeq.from(values.map(_.map(ArraySeq.from).to(ArraySeq)).to(ArraySeq)), DType.Float32)

  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      values: Seq[Seq[Seq[Float]]],
      dtype: DType
  ): Tensor[(L1, L2, L3)] =
    apply(ArraySeq.from(values.map(_.map(ArraySeq.from).to(ArraySeq)).to(ArraySeq)), dtype)

  def fromArray[L1 <: Label, L2 <: Label, L3 <: Label](
      shape: Shape3[L1, L2, L3],
      values: ArraySeq[Float],
      dtype: DType = DType.Float32
  ): Tensor[(L1, L2, L3)] =

    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val valuesPy = values.toArray.toPythonCopy
    val jaxValues = Jax.jnp
      .array(valuesPy, dtype = JaxDType.jaxDtype(dtype))
      .reshape(shape.dims(0), shape.dims(1), shape.dims(2))
    new Tensor(shape, jaxValues, dtype)

class TensorIndexer[T <: Tuple](
    private val tensor: Tensor[T],
    private val index: ToIntTuple[T]
):

  val idxAsSeq: Seq[Int] = index.productIterator.toSeq.asInstanceOf[Seq[Int]]

  def get: Tensor.Tensor0 =
    // Convert sequence to Python tuple for JAX indexing
    val indexTuple = Jax.Dynamic.global.tuple(idxAsSeq.toPythonProxy)

    // Get the value at the specified index
    val atHelper = tensor.jaxValue.at.__getitem__(indexTuple)
    val jaxScalar = atHelper.get()

    // Create a scalar tensor with the retrieved value
    new Tensor[EmptyTuple](Shape.empty, jaxScalar, tensor.dtype)

  def set[U <: Tuple](value: Tensor.Tensor0): Tensor[T] =
    // Get the JAX value of the tensor to set
    val jaxValueToSet = value.jaxValue

    // Convert sequence to Python tuple for JAX indexing
    val indexTuple = Jax.Dynamic.global.tuple(idxAsSeq.toPythonProxy)

    // Use JAX's at[].set() function to set the value at the specified index
    val atHelper = tensor.jaxValue.at.__getitem__(indexTuple)
    val updatedJaxValue = atHelper.set(jaxValueToSet)

    // Create a new tensor with the updated value
    new Tensor(tensor.shape, updatedJaxValue, tensor.dtype)
