package shapeful.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import shapeful.jax.Jax
import shapeful.jax.JaxDType
import shapeful.jax.Jax.PyDynamic
import shapeful.tensor.{Label, Labels}
import shapeful.random.Random
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import scala.reflect.ClassTag
import me.shadaj.scalapy.readwrite.Writer

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  // case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])
  case Other(name: String) extends Device(Jax.Dynamic.global.none)

object Device:
  val default: Device = Device.CPU
  val values: Seq[Device] = Seq(
    Device.CPU
  )

trait ScalarValue[V]:
  def dtype: DType

object ScalarValue:

  given floatValue: ScalarValue[Float] with
    def dtype: DType = DType.Float32
  given intValue: ScalarValue[Int] with
    def dtype: DType = DType.Int32
  given booleanValue: ScalarValue[Boolean] with
    def dtype: DType = DType.Bool

object TensorValue:
  def apply[A: ScalarValue]: TensorValue[A] = new TensorValueImpl[A]()
  def merge[V1, V2](tv1: TensorValue[V1], tv2: TensorValue[V2])(
    using ev: V1 =:= V2
  ): TensorValue[V1] = tv1

sealed trait TensorValue[A: ScalarValue]:
  def dtype: DType = summon[ScalarValue[A]].dtype

class TensorValueImpl[A](using val sv: ScalarValue[A]) extends TensorValue[A]

class Tensor[T <: Tuple: Labels, V] private[tensor](
  val tv: TensorValue[V]
)(
  val jaxValue: Jax.PyDynamic
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)

  lazy val device: Device = Device.values
    .find(d => Jax.device_get(jaxValue).equals(d.jaxDevice))
    .getOrElse(Device.Other(Jax.device_get(jaxValue).name.as[String]))

  def asType(newDType: DType): Tensor[T, V] =
    Tensor(tv).fromPy(Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(newDType)))

  def toDevice(newDevice: Device): Tensor[T, V] =
    Tensor(tv).fromPy(Jax.device_put(jaxValue, newDevice.jaxDevice))

  def equals(other: Tensor[T, V]): Boolean =
    Jax.jnp.array_equal(this.jaxValue, other.jaxValue).item().as[Boolean]

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T, L]): Dim[L] =
    shape.dim(axis)

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [_] =>> Int]

  def apply[V](tv: TensorValue[V]) = new TensorFactory[V](tv)

  class TensorFactory[V](val tv: TensorValue[V]):
    def fromPy[T <: Tuple: Labels](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor(tv)(jaxValue)
    def zeros[T <: Tuple: Labels](shape: Shape[T]): Tensor[T, V] = fromPy(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = tv.dtype.jaxType))
    def ones[T <: Tuple: Labels](shape: Shape[T]): Tensor[T, V] = fromPy(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = tv.dtype.jaxType))
    def randn[T <: Tuple: Labels](shape: Shape[T])(key: Random.Key): Tensor[T, V] = Random.Normal(tv)(key, shape)
    def const[T <: Tuple: Labels](value: V)(shape: Shape[T])(using writer: Writer[V]): Tensor[T, V] = Tensor(tv).fromPy(Jax.jnp.full(shape.dimensions.toPythonProxy, value, dtype = tv.dtype.jaxType))
    def fromArray[T <: Tuple: Labels](shape: Shape[T], values: Array[V])(using 
      py.ConvertableToSeqElem[V],
    ): Tensor[T, V] =
      require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
      val jaxValues = Jax.jnp.array(
        values.toPythonProxy,
        dtype = tv.dtype.jaxType
      ).reshape(shape.dimensions.toPythonProxy)
      fromPy(jaxValues)

type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  class Tensor0Factory[V](tv: TensorValue[V]):
    def zero: Tensor0[V] = Tensor(tv).zeros(Shape.empty)
    def one: Tensor0[V] = Tensor(tv).ones(Shape.empty)
    def randn(key: Random.Key): Tensor0[V] = Tensor(tv).randn(Shape.empty)(key)
    def const(value: V)(using writer: Writer[V]): Tensor0[V] = Tensor(tv).const(value)(Shape.empty)

  def apply[V](tv: TensorValue[V]): Tensor0Factory[V] = Tensor0Factory[V](tv)
  def apply[V](value: V)(using sv: ScalarValue[V], writer: Writer[V]): Tensor0[V] = Tensor(TensorValue[V]).const(value)(Shape.empty)

object Tensor1:

  class Tensor1Factory[V](val tv: TensorValue[V]):
    def zeros[L: Label](dim: Dim[L]): Tensor1[L, V] = Tensor(tv).zeros(Shape1(dim))
    def ones[L: Label](dim: Dim[L]): Tensor1[L, V] = Tensor(tv).ones(Shape1(dim))
    def randn[L: Label](dim: Dim[L])(key: Random.Key): Tensor1[L, V] = Tensor(tv).randn(Shape1(dim))(key)
    def const[L: Label](value: V)(dim: Dim[L])(using writer: Writer[V]): Tensor1[L, V] = Tensor(tv).const(value)(Shape1(dim))
    def fromArray[L: Label](axis: Axis[L], values: Array[V])(
      using py.ConvertableToSeqElem[V]
    ): Tensor1[L, V] = Tensor(tv).fromPy(Jax.jnp.array(
      values.toPythonProxy,
      dtype = tv.dtype.jaxType
    ))

  def apply[V](tv: TensorValue[V]): Tensor1Factory[V] = Tensor1Factory[V](tv)
  
object Tensor2:

  class Tensor2Factory[V](val tv: TensorValue[V]):
    def zeros[L1: Label, L2: Label](dim1: Dim[L1], dim2: Dim[L2]): Tensor2[L1, L2, V] = Tensor(tv).zeros(Shape2(dim1, dim2))
    def ones[L1: Label, L2: Label](dim1: Dim[L1], dim2: Dim[L2]): Tensor2[L1, L2, V] = Tensor(tv).ones(Shape2(dim1, dim2))
    def randn[L1: Label, L2: Label](dim1: Dim[L1], dim2: Dim[L2])(key: Random.Key): Tensor2[L1, L2, V] = Tensor(tv).randn(Shape2(dim1, dim2))(key)
    def const[L1: Label, L2: Label](value: V)(dim1: Dim[L1], dim2: Dim[L2])(using writer: Writer[V]): Tensor2[L1, L2, V] = Tensor(tv).const(value)(Shape2(dim1, dim2))
    def fromArray[L1: Label, L2: Label](
      axis1: Axis[L1],
      axis2: Axis[L2],
      values: Array[Array[V]]
    )(using 
      py.ConvertableToSeqElem[V],
      ClassTag[V],
    ): Tensor2[L1, L2, V] = 
      val dims = (axis1 -> values.length, axis2 -> values.head.length)
      Tensor(tv).fromArray(Shape(dims), values.flatten)

    def eye[L: Label](dim: Dim[L]): Tensor2[L, L, V] = Tensor(tv).fromPy(Jax.jnp.eye(dim._2, dtype = tv.dtype.jaxType))
    def diag[L: Label](diag: Tensor1[L, V]): Tensor2[L, L, V] = Tensor(tv).fromPy(Jax.jnp.diag(diag.jaxValue))

  def apply[V](tv: TensorValue[V]): Tensor2Factory[V] = Tensor2Factory[V](tv)

object Tensor3:

  class Tensor3Factory[V](val tv: TensorValue[V]):
    def zeros[L1: Label, L2: Label, L3: Label](dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3]): Tensor3[L1, L2, L3, V] = Tensor(tv).zeros(Shape3(dim1, dim2, dim3))
    def ones[L1: Label, L2: Label, L3: Label](dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3]): Tensor3[L1, L2, L3, V] = Tensor(tv).ones(Shape3(dim1, dim2, dim3))
    def randn[L1: Label, L2: Label, L3: Label](dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3])(key: Random.Key): Tensor3[L1, L2, L3, V] = Tensor(tv).randn(Shape3(dim1, dim2, dim3))(key)
    def const[L1: Label, L2: Label, L3: Label](value: V)(dim1: Dim[L1], dim2: Dim[L2], dim3: Dim[L3])(using writer: Writer[V]): Tensor3[L1, L2, L3, V] = Tensor(tv).const(value)(Shape3(dim1, dim2, dim3))
  end Tensor3Factory

  def apply[V](tv: TensorValue[V]): Tensor3Factory[V] = Tensor3Factory[V](tv)


type FloatTensor[T <: Tuple] = Tensor[T, Float]
type FloatTensor0 = Tensor0[Float]
type FloatTensor1[L] = Tensor1[L, Float]
type FloatTensor2[L1, L2] = Tensor2[L1, L2, Float]
type FloatTensor3[L1, L2, L3] = Tensor3[L1, L2, L3, Float]

object FloatTensor:
  val factory = Tensor.TensorFactory(TensorValue[Float])
  export factory.*

object FloatTensor0:
  val factory = Tensor0.Tensor0Factory[Float](TensorValue[Float])
  export factory.*

  def apply(value: Float)(using sv: ScalarValue[Float], writer: Writer[Float]): Tensor0[Float] = 
    Tensor(TensorValue[Float]).const(value)(Shape.empty)

object FloatTensor1:
  val factory = Tensor1.Tensor1Factory[Float](TensorValue[Float])
  export factory.*

object FloatTensor2:
  val factory = Tensor2.Tensor2Factory[Float](TensorValue[Float])
  export factory.*

object FloatTensor3:
  val factory = Tensor3.Tensor3Factory[Float](TensorValue[Float])
  export factory.*


type IntTensor[T <: Tuple] = Tensor[T, Int]
type IntTensor0 = Tensor0[Int]
type IntTensor1[L] = Tensor1[L, Int]
type IntTensor2[L1, L2] = Tensor2[L1, L2, Int]
type IntTensor3[L1, L2, L3] = Tensor3[L1, L2, L3, Int]

object IntTensor:
  val factory = Tensor.TensorFactory(TensorValue[Int])
  export factory.*

object IntTensor0:
  val factory = Tensor0.Tensor0Factory[Int](TensorValue[Int])
  export factory.*

  def apply(value: Int)(using sv: ScalarValue[Int], writer: Writer[Int]): Tensor0[Int] = 
    Tensor(TensorValue[Int]).const(value)(Shape.empty)

object IntTensor1:
  val factory = Tensor1.Tensor1Factory[Int](TensorValue[Int])
  export factory.*

object IntTensor2:
  val factory = Tensor2.Tensor2Factory[Int](TensorValue[Int])
  export factory.*

object IntTensor3:
  val factory = Tensor3.Tensor3Factory[Int](TensorValue[Int])
  export factory.*


type BooleanTensor[T <: Tuple] = Tensor[T, Boolean]
type BooleanTensor0 = Tensor0[Boolean]
type BooleanTensor1[L] = Tensor1[L, Boolean]
type BooleanTensor2[L1, L2] = Tensor2[L1, L2, Boolean]
type BooleanTensor3[L1, L2, L3] = Tensor3[L1, L2, L3, Boolean]

object BooleanTensor:
  val factory = Tensor.TensorFactory(TensorValue[Boolean])
  export factory.*

object BooleanTensor0:
  val factory = Tensor0.Tensor0Factory[Boolean](TensorValue[Boolean])
  export factory.*

  def apply(value: Boolean)(using sv: ScalarValue[Boolean], writer: Writer[Boolean]): Tensor0[Boolean] = 
    Tensor(TensorValue[Boolean]).const(value)(Shape.empty)

object BooleanTensor1:
  val factory = Tensor1.Tensor1Factory[Boolean](TensorValue[Boolean])
  export factory.*

object BooleanTensor2:
  val factory = Tensor2.Tensor2Factory[Boolean](TensorValue[Boolean])
  export factory.*

object BooleanTensor3:
  val factory = Tensor3.Tensor3Factory[Boolean](TensorValue[Boolean])
  export factory.*