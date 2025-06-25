package shapeful.jax

import scala.language.experimental.namedTypeArguments
import shapeful.*
import me.shadaj.scalapy.py.PyQuote
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import shapeful.tensor.TupleHelpers
import shapeful.tensor.DType
import shapeful.jax.JaxDType

trait ToPyTree[P]:
  def toPyTree(p: P): Jax.PyAny
  def fromPyTree(p: Jax.PyAny): P

object ToPyTree:

  // Define necessary ToPyTree instances
  given [T <: Tuple]: ToPyTree[Tensor[T]] with
    def toPyTree(t: Tensor[T]): Jax.PyAny = t.jaxValue
    def fromPyTree(p: Jax.PyAny): Tensor[T] =
      val shapeJax = p.as[Jax.PyDynamic].shape.as[Seq[Int]]
      val shapeTuple = TupleHelpers.createTupleFromSeq[T](shapeJax)
      val shape = Shape[T](shapeTuple)
      val dtype = JaxDType.fromJaxDtype(p.as[Jax.PyDynamic].dtype)
      new Tensor[T](shape, p.as[Jax.PyDynamic], dtype)

  // Manual instance for tuple of 2 tensors (arbitrary ranks)
  given [T1 <: Tuple, T2 <: Tuple]: ToPyTree[(Tensor[T1], Tensor[T2])] with
    def toPyTree(t: (Tensor[T1], Tensor[T2])): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(t._1.jaxValue, t._2.jaxValue).toPythonProxy)

    def fromPyTree(p: Jax.PyAny): (Tensor[T1], Tensor[T2]) =
      val pyTuple = p.as[py.Dynamic]
      val tensor1PyTree = pyTuple.bracketAccess(0)
      val tensor2PyTree = pyTuple.bracketAccess(1)

      val tensor1 = summon[ToPyTree[Tensor[T1]]].fromPyTree(tensor1PyTree)
      val tensor2 = summon[ToPyTree[Tensor[T2]]].fromPyTree(tensor2PyTree)

      (tensor1, tensor2)

  // Manual instance for tuple of 3 tensors (arbitrary ranks)
  given [T1 <: Tuple, T2 <: Tuple, T3 <: Tuple]: ToPyTree[(Tensor[T1], Tensor[T2], Tensor[T3])] with
    def toPyTree(t: (Tensor[T1], Tensor[T2], Tensor[T3])): Jax.PyAny =
      py.Dynamic.global.tuple(Seq(t._1.jaxValue, t._2.jaxValue, t._3.jaxValue).toPythonProxy)

    def fromPyTree(p: Jax.PyAny): (Tensor[T1], Tensor[T2], Tensor[T3]) =
      val pyTuple = p.as[py.Dynamic]
      val tensor1PyTree = pyTuple.bracketAccess(0)
      val tensor2PyTree = pyTuple.bracketAccess(1)
      val tensor3PyTree = pyTuple.bracketAccess(2)

      val tensor1 = summon[ToPyTree[Tensor[T1]]].fromPyTree(tensor1PyTree)
      val tensor2 = summon[ToPyTree[Tensor[T2]]].fromPyTree(tensor2PyTree)
      val tensor3 = summon[ToPyTree[Tensor[T3]]].fromPyTree(tensor3PyTree)

      (tensor1, tensor2, tensor3)
