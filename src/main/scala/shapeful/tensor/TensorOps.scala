package shapeful.tensor

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import scala.annotation.targetName
import scala.util.NotGiven

object TensorOps:
  extension [T <: Tuple](t: Tensor[T])

    // Addition with same shape - most common case
    def +(other: Tensor[T])(using NotGiven[T =:= EmptyTuple]): Tensor[T] =
      if t.shape.dims != other.shape.dims then throw new IllegalArgumentException("Shapes must match for addition")
      val result = Jax.jnp.add(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensor0PlusScalar")
    def +(other: Tensor0): Tensor[T] =
      val result = Jax.jnp.add(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    // Subtraction with same shape - most common case
    def -(other: Tensor[T])(using NotGiven[T =:= EmptyTuple]): Tensor[T] =
      if t.shape.dims != other.shape.dims then throw new IllegalArgumentException("Shapes must match for subtraction")
      val result = Jax.jnp.subtract(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensor0MinusScalar")
    def -(other: Tensor0): Tensor[T] =
      val result = Jax.jnp.subtract(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    def *(other: Tensor[T])(using NotGiven[T =:= EmptyTuple]): Tensor[T] =
      if t.shape.dims != other.shape.dims then
        throw new IllegalArgumentException("Shapes must match for multiplication")
      val result = Jax.jnp.multiply(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensor0MultScalar")
    def *(other: Tensor0): Tensor[T] =
      val result = Jax.jnp.multiply(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensorDivTensor")
    def /(other: Tensor[T])(using NotGiven[T =:= EmptyTuple]): Tensor[T] =
      if t.shape.dims != other.shape.dims then throw new IllegalArgumentException("Shapes must match for division")
      val result = Jax.jnp.divide(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensorDivScalar")
    def /(other: Tensor0): Tensor[T] =
      val result = Jax.jnp.divide(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.promoteTypes(t.dtype, other.dtype))

    def exp: Tensor[T] =
      val result = Jax.jnp.exp(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def log: Tensor[T] =
      val result = Jax.jnp.log(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def pow(n: Tensor0): Tensor[T] =
      val result = Jax.jnp.pow(t.jaxValue, n.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def norm: Tensor0 =
      val result = Jax.jnp.linalg.norm(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    // Reduction operations
    def sum: Tensor0 =
      val result = Jax.jnp.sum(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    def mean: Tensor0 =
      val result = Jax.jnp.mean(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    def min: Tensor0 =
      val result = Jax.jnp.min(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    def max: Tensor0 =
      val result = Jax.jnp.max(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    def argmin: Tensor0 =
      val result = Jax.jnp.argmin(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.Int32)

    def argmax: Tensor0 =
      val result = Jax.jnp.argmax(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.Int32)

    def std: Tensor0 =
      val result = Jax.jnp.std(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    def variance: Tensor0 =
      val result = Jax.jnp.`var`(t.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

    // Additional math operations
    def abs: Tensor[T] =
      val result = Jax.jnp.abs(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def sqrt: Tensor[T] =
      val result = Jax.jnp.sqrt(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def sin: Tensor[T] =
      val result = Jax.jnp.sin(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def cos: Tensor[T] =
      val result = Jax.jnp.cos(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    def tanh: Tensor[T] =
      val result = Jax.jnp.tanh(t.jaxValue)
      new Tensor[T](t.shape, result, t.dtype)

    // Comparison operations
    def <(other: Tensor[T]): Tensor[T] =
      val result = Jax.jnp.less(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.Bool)

    def <=(other: Tensor[T]): Tensor[T] =
      val result = Jax.jnp.less_equal(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.Bool)

    def >(other: Tensor[T]): Tensor[T] =
      val result = Jax.jnp.greater(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.Bool)

    def >=(other: Tensor[T]): Tensor[T] =
      val result = Jax.jnp.greater_equal(t.jaxValue, other.jaxValue)
      new Tensor[T](t.shape, result, DType.Bool)

  extension (t: Tensor0)
    def toInt: Int = t.jaxValue.item().as[Int]
    def toFloat: Float = t.jaxValue.item().as[Float]
    def toBool: Boolean = t.jaxValue.item().as[Boolean]

    // Scalar arithmetic operations
    def +(other: Tensor0): Tensor0 =
      val result = Jax.jnp.add(t.jaxValue, other.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.promoteTypes(t.dtype, other.dtype))

    def -(other: Tensor0): Tensor0 =
      val result = Jax.jnp.subtract(t.jaxValue, other.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.promoteTypes(t.dtype, other.dtype))

    def *(other: Tensor0): Tensor0 =
      val result = Jax.jnp.multiply(t.jaxValue, other.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.promoteTypes(t.dtype, other.dtype))

    def /(other: Tensor0): Tensor0 =
      val result = Jax.jnp.divide(t.jaxValue, other.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensor0Pow")
    def pow(exponent: Tensor0): Tensor0 =
      val result = Jax.jnp.pow(t.jaxValue, exponent.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, t.dtype)

  extension [L <: Label](t: Tensor1[L])
    def dot(other: Tensor1[L]): Tensor[EmptyTuple] =
      val result = Jax.jnp.dot(t.jaxValue, other.jaxValue)
      new Tensor[EmptyTuple](Shape.empty, result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensor1MatmulTensor2")
    def matmul[L2 <: Label](other: Tensor2[L, L2]): Tensor1[L2] =
      val result = Jax.jnp.dot(t.jaxValue, other.jaxValue)
      new Tensor(Shape1[L2](other.shape.dim[L2]), result, DType.promoteTypes(t.dtype, other.dtype))

    @targetName("tensor1as")
    def as[NewL <: Label]: Tensor[Tuple1[NewL]] =
      t.relabel[Tuple1[NewL]]

  extension [L1 <: Label, L2 <: Label](t: Tensor2[L1, L2])

    def transpose: Tensor2[L2, L1] =
      val result = Jax.jnp.transpose(t.jaxValue)
      new Tensor(Shape2[L2, L1](t.shape.dim[L2], t.shape.dim[L1]), result, t.dtype)

    @targetName("tensor2MatmulTensor2")
    def matmul[L2Other <: Label](other: Tensor2[L2, L2Other]): Tensor2[L1, L2Other] =
      val result = Jax.jnp.matmul(t.jaxValue, other.jaxValue)
      new Tensor(
        Shape2[L1, L2Other](t.shape.dim[L1], other.shape.dim[L2Other]),
        result,
        DType.promoteTypes(t.dtype, other.dtype)
      )

    @targetName("tensor2MatmulTensor1")
    def matmul1(other: Tensor1[L2]): Tensor1[L1] =
      val result = Jax.jnp.dot(t.jaxValue, other.jaxValue)
      new Tensor(Shape1[L1](t.shape.dim[L1]), result, DType.promoteTypes(t.dtype, other.dtype))

    /** Compute the determinant of a square matrix
      *
      * @return
      *   A scalar tensor containing the determinant
      */
    @targetName("tensor2Det")
    def det: Tensor0 =
      val detValue = Jax.jnp.linalg.det(t.jaxValue)
      new Tensor(Shape0, detValue.as[Jax.PyDynamic], t.dtype)

    /** Compute the inverse of a square matrix
      *
      * @return
      *   The inverse matrix with transposed dimensions
      */
    @targetName("tensor2Inv")
    def inv: Tensor2[L2, L1] =
      shapeful.linalg.Linalg.inverse(t)

    @targetName("tensor2as")
    def as[NewL1 <: Label, NewL2 <: Label]: Tensor[(NewL1, NewL2)] =
      t.relabel[(NewL1, NewL2)]
