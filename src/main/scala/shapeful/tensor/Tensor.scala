package shapeful.tensor

import torch.Int32
import torch.Float32
import torch.DType.float32

import scala.deriving.Mirror

trait Tensor[DType <: torch.DType]: 

  def repr: torch.Tensor[DType]
  def shape: Shape
  def dtype: DType


  def toVariable : Variable =
    val newRepr : torch.Tensor[Float32] = repr.detach().to(float32)
    newRepr.requiresGrad = true
    this match
      case t: Tensor0[DType] => new Variable0(newRepr)
      case t: Tensor1[a, DType] => new Variable1(shape.asInstanceOf[Shape1[a]], newRepr)
      case t: Tensor2[a, b, DType] => new Variable2(shape.asInstanceOf[Shape2[a, b]], newRepr)
      case t: Tensor3[a, b, c, DType] => new Variable3(shape.asInstanceOf[Shape3[a, b, c]], newRepr)
      case t: Tensor4[a, b, c, d, DType] => new Variable4(shape.asInstanceOf[Shape4[a, b, c, d]], newRepr)
      case _ => throw new Exception("Unsupported tensor shape")

  override def toString(): String = 
    repr.toString()

object Tensor:
  def fromTorch[DType <: torch.DType](t: torch.Tensor[DType]): Tensor[DType] =
    t.shape.size match
      case 0 => new Tensor0(t, t.dtype)
      case 1 => new Tensor1(Shape(0 ->> t.shape(0)), t, t.dtype)
      case 2 => new Tensor2(Shape(0 ->> t.shape(0), 1 ->> t.shape(1)), t, t.dtype)
      case 3 => new Tensor3(Shape(0 ->> t.shape(0), 1 ->> t.shape(1), 2 ->> t.shape(2)), t, t.dtype)
      case 4 => new Tensor4(Shape(0 ->> t.shape(0), 1 ->> t.shape(1), 2 ->> t.shape(2), 3 ->> t.shape(3)), t, t.dtype)
      case _ => throw new Exception("Unsupported tensor shape")


  given createfromRepr: FromRepr[Float32, Tensor[Float32]]
  with
    def createfromRepr(repr: torch.Tensor[Float32]): Tensor[Float32] =
      fromTorch(repr)

trait Variable extends Tensor[Float32]
  //override val dtype : Float32 = float32

object Variable:
  def fromTorch(t: torch.Tensor[Float32]): Variable = t match
    case t if t.shape.size == 0 => new Variable0(t)
    case t if t.shape.size == 1 => new Variable1(Shape(0 ->> t.shape(0)), t)
    case t if t.shape.size == 2 => new Variable2(Shape(0 ->> t.shape(0), 1 ->> t.shape(1)), t)
    case _ => throw new Exception("Unsupported tensor shape")

  given createfromRepr: FromRepr[Float32, Variable]
  with
    def createfromRepr(repr: torch.Tensor[Float32]): Variable =
      fromTorch(repr)


class Tensor0[DType <: torch.DType](override val repr: torch.Tensor[DType], override val dtype: DType)
    extends Tensor[DType]:

  private val dummy = repr.info // hack to force initialization of the tensor

  def item: torch.DTypeToScala[DType] = repr.item
  override val shape = Shape0

object Tensor0:
  def apply[DType <: torch.DType](
      value: Float,
      dtype: DType = float32
  ): Tensor0[DType] =
    val stensor = torch.Tensor(value).to[DType](dtype)
    stensor.requiresGrad = false
    new Tensor0(stensor, dtype)

  given createfromRepr[DType <: torch.DType]: FromRepr[DType, Tensor0[DType]]
  with
    def createfromRepr(repr: torch.Tensor[DType]): Tensor0[DType] =
      new Tensor0[DType](repr, repr.dtype)

  given createfromReprFloat32: FromRepr[Float32, Tensor0[Float32]] with
    def createfromRepr(repr: torch.Tensor[Float32]): Tensor0[Float32] =
      new Tensor0[Float32](repr, repr.dtype)

class Variable0(repr: torch.Tensor[Float32]) extends Tensor0[Float32](repr, float32) with Variable
object Variable0:
  def apply[A, b](value: Float) =
    val stensor = torch.Tensor(value)
    stensor.requiresGrad = true
    new Variable0(stensor)


  given createfromRep: FromRepr[Float32, Variable0]
  with
    def createfromRepr(repr: torch.Tensor[Float32]): Variable0 =
      new Variable0(repr)

class Tensor1[A <: Singleton, DType <: torch.DType](
    override val shape: Shape1[A],
    override val repr: torch.Tensor[DType],
    override val dtype: DType
) extends Tensor[DType]:


  private val s = repr.info // hack to force initialization of the tensor

  def apply(i: Int): Tensor0[DType] =
    new Tensor0[DType](repr(i), dtype)

object Tensor1:
  def apply[A <: Singleton,DType <: torch.DType](
      shape: Shape1[A],
      initializer: => Float,
  ): Tensor1[A, Float32] =

    apply(shape, initializer, float32)

  def apply[A <: Singleton,DType <: torch.DType](
      shape: Shape1[A],
      initializer: => Float,
      dtype: DType
  ): Tensor1[A, DType] =
    val data = Array.fill(shape.dim1)(initializer)
    val sTensor = torch.Tensor(data).reshape(shape.dim1).to(dtype)
    new Tensor1[A, DType](shape, sTensor, dtype)


  def fromSeq[A <: Singleton, DType <: torch.DType](
      shape: Shape1[A],
      data: Seq[Float]
  ): Tensor1[A, Float32] =
    val sTensor = torch.Tensor(data.toArray).reshape(shape.dim1)
    new Tensor1[A, Float32](shape, sTensor, float32)

  given createfromRepr[DType <: torch.DType, A <: Singleton]
      : FromRepr[DType, Tensor1[A, DType]] with
    def createfromRepr(repr: torch.Tensor[DType]): Tensor1[A, DType] =
      new Tensor1[A, DType](new Shape1[A](repr.shape(0)), repr, repr.dtype)

class Variable1[A <: Singleton](shape: Shape1[A], repr: torch.Tensor[Float32])
    extends Tensor1[A, Float32](shape, repr, dtype=float32) with Variable:
    repr.requiresGrad = true

object Variable1:
  def apply[A <: Singleton](
      shape: Shape1[A],
      initializer: => Float
  ): Variable1[A] =
    val data = Array.fill(shape.dim1)(initializer)
    val sTensor = torch.Tensor(data).reshape(shape.dim1)
    sTensor.requiresGrad = true
    new Variable1[A](shape, sTensor)


  given createfromRepr[A <: Singleton]
      : FromRepr[Float32, Variable1[A]] with
    def createfromRepr(repr: torch.Tensor[Float32]): Variable1[A] =
      new Variable1[A](new Shape1[A](repr.shape(0)), repr)


/** A constant is a tensor that does not need parameters
  */
class Tensor2[A <: Singleton, B <: Singleton, DType <: torch.DType](
    override val shape: Shape2[A, B],
    override val repr: torch.Tensor[DType], 
    override val dtype: DType = float32
) extends Tensor[DType]:

  private val s = repr.info // hack to force initialization of the tensor
  def apply(i: Int, j: Int): Tensor0[DType] =
    new Tensor0[DType](repr(i, j), dtype)

  def update(i: Int, j: Int, value: Tensor0[DType]): Unit =
    repr.update(Seq(i, j), value.repr)



  def to[C <: Singleton, D <: Singleton, DType <: torch.DType](dtype : DType): Tensor2[C, D, DType] =
    val newShape = new Shape2[C, D](shape.dim1, shape.dim2)
    new Tensor2[C, D, DType](newShape, repr.to(dtype), dtype)

object Tensor2:

  def apply[A <: Singleton, B <: Singleton](
      shape: Shape2[A, B],
      initializer: => Float,
  ): Tensor2[A, B, Float32] =
    apply(shape, initializer, float32)

  def apply[A <: Singleton, B <: Singleton, DType <: torch.DType](
      shape: Shape2[A, B],
      initializer: => Float,
      dtype: DType
  ): Tensor2[A, B, DType] =

    val data = Array.fill(shape.dim1 * shape.dim2)(initializer)
    val sTensor = torch.Tensor(data).reshape(shape.dim1, shape.dim2).to(dtype)
    new Tensor2[A, B, DType](shape, sTensor, dtype)


  def eye[A <: Singleton, B <: Singleton](
      shape: Shape2[A, B]
  ): Tensor2[A, B, Float32] =
    val sTensor = torch.eye(shape.dim1, None)
    new Tensor2[A, B, Float32](shape, sTensor, float32)


  def fromSeq[A <: Singleton, B <: Singleton](
      shape: Shape2[A, B],
      data: Seq[Float]
  ): Tensor2[A, B, Float32] =
    val sTensor = torch.Tensor(data.toArray).reshape(shape.dim1, shape.dim2)
    new Tensor2[A, B, Float32](shape, sTensor, float32)

  given createfromRepr[DType <: torch.DType, A <: Singleton, B <: Singleton]
      : FromRepr[DType, Tensor2[A, B, DType]] with
    def createfromRepr(repr: torch.Tensor[DType]): Tensor2[A, B, DType] =
      new Tensor2[A, B, DType](
        new Shape2[A, B](repr.shape(0), repr.shape(1)),
        repr, 
        repr.dtype
      )

/** A variable is a tensor that needs parameters
  */
class Variable2[A <: Singleton, B <: Singleton](shape: Shape2[A, B], repr: torch.Tensor[Float32])
    extends Tensor2[A, B, Float32](shape, repr) with Variable
object Variable2:
  def apply[A <: Singleton, B <: Singleton](
      shape: Shape2[A, B],
      initializer: => Float
  ): Variable2[A, B] =

    val data = Array.fill(shape.dim1 * shape.dim2)(initializer)
    val sTensor = torch.Tensor(data).reshape(shape.dim1, shape.dim2)
    sTensor.requiresGrad = true
    new Variable2[A, B](shape, sTensor)



class Tensor3[A <: Singleton, B <: Singleton, C <: Singleton, DType <: torch.DType](
    override val shape: Shape3[A, B, C],
    override val repr: torch.Tensor[DType], 
    override val dtype: DType = float32
) extends Tensor[DType]:

  private val s = repr.info // hack to force initialization of the tensor

  def apply(i: Int, j: Int, k : Int): Tensor0[DType] =
    new Tensor0[DType](repr(i, j, k), dtype)

  def update(i: Int, j: Int, k : Int, value: Tensor0[DType]): Unit =
    repr.update(Seq(i, j, k), value.repr)



  def to[ANew <: Singleton, BNew <: Singleton, CNew <: Singleton]: Tensor3[ANew, BNew, CNew, DType] =
    val newShape = new Shape3[ANew, BNew, CNew](shape.dim1, shape.dim2, shape.dim3)
    new Tensor3[ANew, BNew, CNew, DType](newShape, repr, dtype)


object Tensor3:
  
    def apply[A <: Singleton, B <: Singleton, C <: Singleton](
        shape: Shape3[A, B, C],
        initializer: => Float,
    ): Tensor3[A, B, C, Float32] =
      val data = Array.fill(shape.dim1 * shape.dim2 * shape.dim3)(initializer)
      val repr = torch.Tensor(data).reshape(shape.dims*)
      new Tensor3(shape, repr, float32)
  
    def fromSeq[A <: Singleton, B <: Singleton, C <: Singleton](
        shape: Shape3[A, B, C],
        data: Seq[Float]
    ): Tensor3[A, B, C, Float32] =
      val sTensor = torch.Tensor(data.toArray).reshape(shape.dims*)
      new Tensor3(shape, sTensor, float32)


    given createfromRepr[DType <: torch.DType, A <: Singleton, B <: Singleton, C <: Singleton]
      : FromRepr[DType, Tensor3[A, B, C, DType]] with
        def createfromRepr(repr: torch.Tensor[DType]): Tensor3[A, B, C, DType] =
          new Tensor3[A, B, C, DType](
            new Shape3[A, B, C](repr.shape(0), repr.shape(1), repr.shape(2)),
            repr, 
            repr.dtype
      )



class Variable3[A <: Singleton, B <: Singleton, C <: Singleton](shape: Shape3[A, B, C], repr: torch.Tensor[Float32])
    extends Tensor3[A, B, C, Float32](shape, repr) with Variable
object Variable3:
  def apply[A <: Singleton, B <: Singleton, C <: Singleton](
      shape: Shape3[A, B, C],
      initializer: => Float
  ): Variable3[A, B, C] =

    val data = Array.fill(shape.dim1 * shape.dim2 * shape.dim3)(initializer)
    val sTensor = torch.Tensor(data).reshape(shape.dim1, shape.dim2, shape.dim3)
    sTensor.requiresGrad = true
    new Variable3[A, B, C](shape, sTensor)



class Tensor4[A <: Singleton, B <: Singleton, C <: Singleton, D <: Singleton, DType <: torch.DType](
    override val shape: Shape4[A, B, C, D],
    override val repr: torch.Tensor[DType], 
    override val dtype: DType = float32
) extends Tensor[DType]:

  private val s = repr.info // hack to force initialization of the tensor

  def apply(i: Int, j: Int, k : Int, l : Int): Tensor0[DType] =
    new Tensor0[DType](repr(i, j, k, l), dtype)

  def update(i: Int, j: Int, k : Int, l : Int,  value: Tensor0[DType]): Unit =
    repr.update(Seq(i, j, k, l), value.repr)



object Tensor4:
 
    def apply[A <: Singleton, B <: Singleton, C <: Singleton, D <: Singleton](
        shape: Shape4[A, B, C, D],
        initializer: => Float,
    ): Tensor4[A, B, C, D, Float32] =
      val data = Array.fill(shape.dim1 * shape.dim2 * shape.dim3 * shape.dim4)(initializer)
      val repr = torch.Tensor(data).reshape(shape.dims*)
      new Tensor4(shape, repr, float32)
  
    def fromSeq[A <: Singleton, B <: Singleton, C <: Singleton, D <: Singleton](
        shape: Shape4[A, B, C, D],
        data: Seq[Float]
    ): Tensor4[A, B, C, D, Float32] =
      val sTensor = torch.Tensor(data.toArray).reshape(shape.dims*)
      new Tensor4(shape, sTensor, float32)


    given createfromRepr[DType <: torch.DType, A <: Singleton, B <: Singleton, C <: Singleton, D <: Singleton]
      : FromRepr[DType, Tensor4[A, B, C, D, DType]] with
        def createfromRepr(repr: torch.Tensor[DType]): Tensor4[A, B, C, D, DType] =
          new Tensor4[A, B, C, D, DType](
            new Shape4[A, B, C, D](repr.shape(0), repr.shape(1), repr.shape(2), repr.shape(3)),
            repr, 
            repr.dtype
      )

class Variable4[A <: Singleton, B <: Singleton, C <: Singleton, D <: Singleton](shape: Shape4[A, B, C, D], repr: torch.Tensor[Float32])
    extends Tensor4[A, B, C, D, Float32](shape, repr) with Variable
object Variable4:
  def apply[A <: Singleton, B <: Singleton, C <: Singleton, D <: Singleton](
      shape: Shape4[A, B, C, D],
      initializer: => Float
  ): Variable4[A, B, C, D] =

    val data = Array.fill(shape.dim1 * shape.dim2 * shape.dim3 * shape.dim4)(initializer)
    val sTensor = torch.Tensor(data).reshape(shape.dim1, shape.dim2, shape.dim3, shape.dim4)
    sTensor.requiresGrad = true
    new Variable4[A, B, C, D](shape, sTensor)


