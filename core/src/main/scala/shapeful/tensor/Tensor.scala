package shapeful.tensor

import scala.annotation.targetName
import scala.compiletime.{erasedValue, summonFrom}
import shapeful.jax.Jax
import shapeful.jax.JaxDType
import shapeful.jax.Jax.PyDynamic
import shapeful.tensor.{Label, Labels}
import shapeful.random.Random
import me.shadaj.scalapy.py.SeqConverters

enum Device(val jaxDevice: PyDynamic):
  case CPU extends Device(Jax.devices("cpu").head.as[PyDynamic])
  // case GPU extends Device(Jax.devices("gpu").head.as[PyDynamic])
  case Other(name: String) extends Device(Jax.Dynamic.global.none)

object Device:
  val default: Device = Device.CPU
  val values: Seq[Device] = Seq(
    Device.CPU
  )

trait Value[V]:
  def dtype: DType

object Value:
  // Singleton value objects for use with Value[X] syntax
  object Float32 extends Value[Float32.type]:
    def dtype: DType = DType.Float32
  
  object Float64 extends Value[Float64.type]:
    def dtype: DType = DType.Float64
  
  object Int32 extends Value[Int32.type]:
    def dtype: DType = DType.Int32
  
  object Int64 extends Value[Int64.type]:
    def dtype: DType = DType.Int64
  
  object Bool extends Value[Bool.type]:
    def dtype: DType = DType.Bool

  // Apply method to enable Value[X] pattern extraction
  def apply[V](value: Value[V]): ValueBuilder[V] = new ValueBuilder[V](value)

  // For custom value types, users should provide a given Value[MyType] with the appropriate dtype
  // Example:
  // trait MyFloatType
  // given Value[MyFloatType] with
  //   def dtype: DType = DType.Float32

class ValueBuilder[V](val evidence: Value[V]):
  def zeros[T <: Tuple : Labels](shape: Shape[T])(using Value[V]): Tensor[T, V] =
    Tensor(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = evidence.dtype.jaxType))(using summon[Labels[T]], summon[Value[V]])
  
  def ones[T <: Tuple : Labels](shape: Shape[T])(using Value[V]): Tensor[T, V] =
    Tensor(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = evidence.dtype.jaxType))(using summon[Labels[T]], summon[Value[V]])
  
  def randn[T <: Tuple : Labels](shape: Shape[T], key: Random.Key)(using Value[V]): Tensor[T, V] =
    Random.normal(key, shape, dtype = evidence.dtype)(using summon[Labels[T]], summon[Value[V]])

class Tensor[T <: Tuple : Labels, V : Value] private[tensor](
  val jaxValue: Jax.PyDynamic,
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)

  lazy val device: Device = Device.values.find(
    d => Jax.device_get(jaxValue).equals(d.jaxDevice)
  ).getOrElse(Device.Other(Jax.device_get(jaxValue).name.as[String]))

  def asType(newDType: DType): Tensor[T, V] = 
    Tensor(jaxValue = Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(newDType)))

  def toDevice(newDevice: Device): Tensor[T, V] = 
    Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice))

  def equals(other: Tensor[T, V]): Boolean =
    Jax.jnp.array_equal(this.jaxValue, other.jaxValue).item().as[Boolean]

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T, L]): Dim[L] = 
    shape.dim(axis)

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [ _ ] =>> Int]

  private[tensor] def apply[T <: Tuple : Labels, V : Value](jaxValue: Jax.PyDynamic): Tensor[T, V] = new Tensor[T, V](jaxValue)

  def fromPy[T <: Tuple : Labels, V : Value](jaxValue: Jax.PyDynamic): Tensor[T, V] = Tensor(jaxValue)

  // Main API: Tensor.of(Value[X]).ones(...)
  def of[V : Value]: TensorFactory[V] = new TensorFactory[V]()

class TensorFactory[V : Value]:
  def zeros[T <: Tuple : Labels](shape: Shape[T]): Tensor[T, V] =
    Tensor(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = summon[Value[V]].dtype.jaxType))
  
  def ones[T <: Tuple : Labels](shape: Shape[T]): Tensor[T, V] =
    Tensor(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = summon[Value[V]].dtype.jaxType))
  
  def randn[T <: Tuple : Labels](shape: Shape[T], key: Random.Key): Tensor[T, V] =
    Random.normal(key, shape, dtype = summon[Value[V]].dtype)(using summon[Labels[T]], summon[Value[V]])
  
  def apply[T <: Tuple : Labels](shape: Shape[T], values: Array[Float], device: Device = Device.default): Tensor[T, V] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val jaxValues = Jax.jnp
      .array(
        values.toPythonProxy,
        dtype = summon[Value[V]].dtype.jaxType,
        device = device.jaxDevice,
      )
      .reshape(shape.dimensions.toPythonProxy)
    Tensor(jaxValues)


type Tensor0[V] = Tensor[EmptyTuple, V]
type Tensor1[L, V] = Tensor[Tuple1[L], V]
type Tensor2[L1, L2, V] = Tensor[(L1, L2), V]
type Tensor3[L1, L2, L3, V] = Tensor[(L1, L2, L3), V]
type Tensor4[L1, L2, L3, L4, V] = Tensor[(L1, L2, L3, L4), V]

object Tensor0:

  // implicit conversions for easy creation
  // TODO move them back
  // given Conversion[Float, Tensor0[Float]] = (x: Float) => Tensor0(x)
  // given Conversion[Int, Tensor0[Int]] = (x: Int) => Tensor0(x)

  def apply[V : Value](jaxValue: Jax.PyDynamic): Tensor0[V] = Tensor(jaxValue)

  def apply[V : Value](value: Float | Int | Boolean): Tensor0[V] =
    value match
      case v: Float   => Tensor0(Jax.jnp.array(v, dtype=DType.Float32.jaxType))
      case v: Int     => Tensor0(Jax.jnp.array(v, dtype=DType.Int32.jaxType))
      case v: Boolean => Tensor0(Jax.jnp.array(v, dtype=DType.Bool.jaxType))

object Tensor1:

  def apply[L : Label, V : Value](axis: Axis[L], values: Array[Float], dtype: DType = DType.Float32): Tensor1[L, V] =
    Tensor(Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

  def fromInts[L : Label, V : Value](axis: Axis[L], values: Array[Int], dtype: DType = DType.Int32): Tensor1[L, V] =
    Tensor(Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

object Tensor2:

  def apply[L1 : Label, L2 : Label, V : Value](
      shape: Shape2[L1, L2],
      values: Array[Float],
      dtype: DType,
  ): Tensor2[L1, L2, V] = 
    val jaxValues = Jax.jnp
      .array(values.toPythonProxy, dtype = dtype.jaxType)
      .reshape(shape.dimensions.toPythonProxy)
    Tensor(jaxValues)

  def apply[L1 : Label, L2 : Label, V : Value](
      shape: Shape2[L1, L2],
      values: Array[Float],
  ): Tensor2[L1, L2, V] = Tensor2(shape, values, summon[Value[V]].dtype)

  def apply[L1 : Label, L2 : Label, V : Value](
      axis1: Axis[L1],
      axis2: Axis[L2],
      values: Array[Array[Float]],
      dtype: DType = DType.Float32,
  ): Tensor[(L1, L2), V] =
    val rows = values.length
    val cols = values.headOption.map(_.length).getOrElse(0)
    require(values.forall(_.length == cols), "All rows must have the same length")
    Tensor2(Shape(axis1 -> rows, axis2 -> cols), values.flatten, dtype)

  def eye[L : Label, V : Value](axis: Axis[L])(dim: Int, dtype: DType = DType.Float32): Tensor2[L, L, V] = 
    Tensor(Jax.jnp.eye(dim, dtype = dtype.jaxType))

  def diag[L : Label, V : Value](diag: Tensor1[L, V]): Tensor2[L, L, V] =
    Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  def apply[L1 : Label, L2 : Label, L3 : Label, V : Value ](
      shape: Shape3[L1, L2, L3],
      values: Array[Float],
      dtype: DType,
  ): Tensor3[L1, L2, L3, V] = 
    val jaxValues = Jax.jnp
      .array(values.toPythonProxy, dtype = dtype.jaxType)
      .reshape(shape.dimensions.toPythonProxy)
    Tensor(jaxValues)

  def apply[L1 : Label, L2 : Label, L3 : Label, V : Value](
      shape: Shape3[L1, L2, L3],
      values: Array[Float],
  ): Tensor3[L1, L2, L3, V] = Tensor3(shape, values, DType.Float32)

  def apply[L1 : Label, L2 : Label, L3 : Label, V : Value](
      axis1: Axis[L1],
      axis2: Axis[L2],
      axis3: Axis[L3],
      values: Array[Array[Array[Float]]],
      dtype: DType = DType.Float32,
  ): Tensor3[L1, L2, L3, V] =
    val dim1 = values.length
    val dim2 = values.headOption.map(_.length).getOrElse(0)
    val dim3 = values.headOption.flatMap(_.headOption).map(_.length).getOrElse(0)
    require(values.forall(_.length == dim2), "All second dimensions must match")
    require(values.forall(_.forall(_.length == dim3)), "All third dimensions must match")
    Tensor3(
      Shape(axis1 -> dim1, axis2 -> dim2, axis3 -> dim3),
      values.flatten.flatten,
      dtype,
    )
