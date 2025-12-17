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

class Tensor[T <: Tuple : Labels] private[tensor](
  val jaxValue: Jax.PyDynamic,
):

  lazy val axes: List[String] = shape.labels
  lazy val dtype: DType = JaxDType.fromJaxDtype(jaxValue.dtype)
  lazy val shape: Shape[T] = Shape.fromList[T](jaxValue.shape.as[Seq[Int]].toList)

  lazy val device: Device = Device.values.find(
    d => Jax.device_get(jaxValue).equals(d.jaxDevice)
  ).getOrElse(Device.Other(Jax.device_get(jaxValue).name.as[String]))

  def asType(newDType: DType): Tensor[T] = 
    Tensor(jaxValue = Jax.jnp.astype(jaxValue, JaxDType.jaxDtype(newDType)))

  def toDevice(newDevice: Device): Tensor[T] = 
    Tensor(jaxValue = Jax.device_put(jaxValue, newDevice.jaxDevice))

  def equals(other: Tensor[T]): Boolean =
    Jax.jnp.array_equal(this.jaxValue, other.jaxValue).item().as[Boolean]

  override def hashCode(): Int = jaxArray.tobytes().hashCode()

  override def toString: String = jaxArray.toString()

  private def jaxArray: Jax.PyDynamic = jaxValue.block_until_ready()

  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T, L]): Dim[L] = 
    shape.dim(axis)

object Tensor:

  type IndicesOf[T <: Tuple] = Tuple.Map[T, [ _ ] =>> Int]

  private[tensor] def apply[T <: Tuple : Labels](jaxValue: Jax.PyDynamic): Tensor[T] = new Tensor[T](jaxValue)

  def fromPy[T <: Tuple : Labels](jaxValue: Jax.PyDynamic): Tensor[T] = Tensor(jaxValue)

  def apply[T <: Tuple : Labels](shape: Shape[T])(using initTensor: Shape[T] => Tensor[T]): Tensor[T] = 
    initTensor(shape)

  def apply[T <: Tuple : Labels](shape: Shape[T], values: Array[Float], dtype: DType = DType.Float32, device: Device = Device.default): Tensor[T] =
    require(values.length == shape.size, s"Values length ${values.length} does not match shape size ${shape.size}")
    val jaxValues = Jax.jnp
      .array(
        values.toPythonProxy,
        dtype = dtype.jaxType,
        device = device.jaxDevice,
      )
      .reshape(shape.dimensions.toPythonProxy)
    Tensor(jaxValues)

  def zeros[T <: Tuple : Labels](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    Tensor(Jax.jnp.zeros(shape.dimensions.toPythonProxy, dtype = dtype.jaxType))

  def ones[T <: Tuple : Labels](shape: Shape[T], dtype: DType = DType.Float32): Tensor[T] =
    Tensor(Jax.jnp.ones(shape.dimensions.toPythonProxy, dtype = dtype.jaxType))

  def randn[T <: Tuple : Labels](shape: Shape[T], key: Random.Key, dtype: DType = DType.Float32): Tensor[T] =
    Random.normal(key, shape, dtype = dtype)

type Tensor0 = Tensor[EmptyTuple]
type Tensor1[L] = Tensor[Tuple1[L]]
type Tensor2[L1, L2] = Tensor[(L1, L2)]
type Tensor3[L1, L2, L3] = Tensor[(L1, L2, L3)]
type Tensor4[L1, L2, L3, L4] = Tensor[(L1, L2, L3, L4)]

object Tensor0:

  // implicit conversions for easy creation
  given Conversion[Float, Tensor0] = (x: Float) => Tensor0(x)
  given Conversion[Int, Tensor0] = (x: Int) => Tensor0(x)

  def apply(jaxValue: Jax.PyDynamic): Tensor0 = Tensor(jaxValue)

  def apply(value: Float | Int | Boolean): Tensor0 =
    value match
      case v: Float   => Tensor0(Jax.jnp.array(v, dtype=DType.Float32.jaxType))
      case v: Int     => Tensor0(Jax.jnp.array(v, dtype=DType.Int32.jaxType))
      case v: Boolean => Tensor0(Jax.jnp.array(v, dtype=DType.Bool.jaxType))

object Tensor1:

  def apply[L : Label](axis: Axis[L], values: Array[Float], dtype: DType = DType.Float32): Tensor1[L] =
    Tensor(Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

  def fromInts[L : Label](axis: Axis[L], values: Array[Int], dtype: DType = DType.Int32): Tensor1[L] =
    Tensor(Jax.jnp.array(values.toPythonProxy, dtype = dtype.jaxType))

object Tensor2:

  def apply[L1 : Label, L2 : Label](
      shape: Shape2[L1, L2],
      values: Array[Float],
      dtype: DType,
  ): Tensor2[L1, L2] = Tensor(shape, values, dtype)

  def apply[L1 : Label, L2 : Label](
      shape: Shape2[L1, L2],
      values: Array[Float],
  ): Tensor[(L1, L2)] = Tensor2(shape, values, DType.Float32)

  def apply[L1 : Label, L2 : Label](
      axis1: Axis[L1],
      axis2: Axis[L2],
      values: Array[Array[Float]],
      dtype: DType = DType.Float32,
  ): Tensor[(L1, L2)] =
    val rows = values.length
    val cols = values.headOption.map(_.length).getOrElse(0)
    require(values.forall(_.length == cols), "All rows must have the same length")
    Tensor2(Shape(axis1 -> rows, axis2 -> cols), values.flatten, dtype)

  def eye[L : Label](axis: Axis[L])(dim: Int, dtype: DType = DType.Float32): Tensor2[L, L] = 
    Tensor(Jax.jnp.eye(dim, dtype = dtype.jaxType))

  def diag[L : Label](diag: Tensor1[L]): Tensor2[L, L] =
    Tensor(Jax.jnp.diag(diag.jaxValue))

object Tensor3:

  def apply[L1 : Label, L2 : Label, L3 : Label](
      shape: Shape3[L1, L2, L3],
      values: Array[Float],
      dtype: DType,
  ): Tensor3[L1, L2, L3] = Tensor(shape, values, dtype)

  def apply[L1 : Label, L2 : Label, L3 : Label](
      shape: Shape3[L1, L2, L3],
      values: Array[Float],
  ): Tensor3[L1, L2, L3] = Tensor3(shape, values, DType.Float32)

  def apply[L1 : Label, L2 : Label, L3 : Label](
      axis1: Axis[L1],
      axis2: Axis[L2],
      axis3: Axis[L3],
      values: Array[Array[Array[Float]]],
      dtype: DType = DType.Float32,
  ): Tensor3[L1, L2, L3] =
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
